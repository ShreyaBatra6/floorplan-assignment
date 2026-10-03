"""Reader for Stray Scanner exports (the LiDAR-tier capture format).

Layout of one capture folder::

    camera_matrix.csv      3x3 intrinsics of the RGB stream (1920x1440)
    odometry.csv           timestamp, frame, x, y, z, qx, qy, qz, qw[, fx, fy, cx, cy, ...]
    imu.csv                accelerometer + gyroscope (unused by the geometry)
    rgb.mp4                RGB video, one video frame per odometry row
    depth/000000.png       uint16 depth in millimetres, 256x192
    confidence/000000.png  uint8 ARKit depth confidence: 0 low, 1 medium, 2 high

Copies are sometimes incomplete (an interrupted transfer leaves fewer depth PNGs than odometry
rows); frames without depth are simply not used for geometry and the gap is reported.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from groundplan.geometry.transforms import pose_matrix

RGB_SIZE = (1920, 1440)


class CaptureError(RuntimeError):
    """The input cannot be processed at the requested tier; the message says what to do."""


@dataclass
class StrayCapture:
    root: Path
    frame_ids: np.ndarray  # (N,) int, the frame index used in file names
    timestamps: np.ndarray  # (N,) seconds
    T_wc: np.ndarray  # (N, 4, 4) OpenCV camera -> world (ARKit, y up)
    K_rgb: np.ndarray  # (N, 3, 3) intrinsics of the RGB image
    depth_size: tuple[int, int] = (256, 192)
    rgb_size: tuple[int, int] = RGB_SIZE
    has_depth: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    has_confidence: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    video_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frame_ids)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if self.n > 1 else 0.0

    def depth_path(self, i: int) -> Path:
        return self.root / "depth" / f"{int(self.frame_ids[i]):06d}.png"

    def confidence_path(self, i: int) -> Path:
        return self.root / "confidence" / f"{int(self.frame_ids[i]):06d}.png"

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres (float32, H x W); 0 where invalid."""
        raw = cv2.imread(str(self.depth_path(i)), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise CaptureError(f"cannot read {self.depth_path(i)}")
        return raw.astype(np.float32) * 1e-3

    def confidence(self, i: int) -> np.ndarray:
        if not self.has_confidence[i]:
            return np.full(self.depth_size[::-1], 2, np.uint8)
        raw = cv2.imread(str(self.confidence_path(i)), cv2.IMREAD_UNCHANGED)
        return raw if raw is not None else np.full(self.depth_size[::-1], 2, np.uint8)

    def K_depth(self, i: int) -> np.ndarray:
        """Intrinsics for the depth map, in depth-pixel-centre coordinates.

        A depth pixel ``u`` covers RGB pixels ``[u/s, (u+1)/s)``, so its centre sits at RGB
        coordinate ``(u + 0.5)/s - 0.5``; folding that into K keeps back-projection exact.
        """
        s = self.depth_size[0] / self.rgb_size[0]
        K = self.K_rgb[i].copy()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        return K

    def load_depth_batch(self, idx: Iterable[int], workers: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
        idx = list(idx)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            depths = list(pool.map(self.depth, idx))
            confs = list(pool.map(self.confidence, idx))
        return list(zip(depths, confs))

    def rgb_frames(self, indices: Iterable[int], max_side: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, BGR image) for the requested capture indices, decoding sequentially."""
        if self.video_path is None:
            return
        wanted = sorted(set(int(i) for i in indices))
        yield from decode_frames(self.video_path, wanted, max_side=max_side)


def decode_frames(video: Path, wanted: list[int], max_side: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the given frame numbers (sorted) from a video, keeping only those frames."""
    if not wanted:
        return
    targets = set(wanted)
    last = wanted[-1]
    try:
        import av

        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for n, frame in enumerate(container.decode(stream)):
                if n in targets:
                    img = frame.to_ndarray(format="bgr24")
                    yield n, _resize(img, max_side)
                if n >= last:
                    break
        return
    except ImportError:  # pragma: no cover - PyAV is a core dependency
        pass
    cap = cv2.VideoCapture(str(video))
    n = 0
    while n <= last:
        ok = cap.grab()
        if not ok:
            break
        if n in targets:
            ok, img = cap.retrieve()
            if ok:
                yield n, _resize(img, max_side)
        n += 1
    cap.release()


def _resize(img: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side or max(img.shape[:2]) <= max_side:
        return img
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)


def _float(x: str) -> float | None:
    x = x.strip()
    return float(x) if x else None


def is_stray_capture(path: Path) -> bool:
    return path.is_dir() and (path / "odometry.csv").exists() and (path / "camera_matrix.csv").exists()


def find_stray_root(path: Path) -> Path | None:
    """The capture folder itself, or a single capture folder nested one or two levels down."""
    if is_stray_capture(path):
        return path
    if path.is_dir():
        hits = [p.parent for p in path.glob("*/odometry.csv")] + [p.parent for p in path.glob("*/*/odometry.csv")]
        hits = [h for h in hits if is_stray_capture(h)]
        if len(hits) == 1:
            return hits[0]
    return None


def load_stray(path: Path, require_depth: bool = True) -> StrayCapture:
    root = find_stray_root(Path(path))
    if root is None:
        raise CaptureError(f"{path} is not a Stray Scanner export (no odometry.csv + camera_matrix.csv)")

    K0 = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    with open(root / "odometry.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    col = {name: k for k, name in enumerate(header)}
    has_intr = all(c in col for c in ("fx", "fy", "cx", "cy"))

    frame_ids, stamps, poses, Ks = [], [], [], []
    for r in rows[1:]:
        if len(r) < 9:
            continue
        frame_ids.append(int(float(r[col.get("frame", 1)])))
        stamps.append(float(r[col.get("timestamp", 0)]))
        pos = np.array([float(r[col["x"]]), float(r[col["y"]]), float(r[col["z"]])])
        quat = np.array([float(r[col["qx"]]), float(r[col["qy"]]), float(r[col["qz"]]), float(r[col["qw"]])])
        poses.append(pose_matrix(pos, quat))
        K = K0.copy()
        if has_intr:
            fx, fy, cx, cy = (_float(r[col[c]]) for c in ("fx", "fy", "cx", "cy"))
            if None not in (fx, fy, cx, cy):
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        Ks.append(K)

    cap = StrayCapture(
        root=root,
        frame_ids=np.array(frame_ids, dtype=int),
        timestamps=np.array(stamps, dtype=float),
        T_wc=np.stack(poses),
        K_rgb=np.stack(Ks),
    )
    video = root / "rgb.mp4"
    cap.video_path = video if video.exists() else None

    depth_dir, conf_dir = root / "depth", root / "confidence"
    depth_names = {p.stem for p in depth_dir.glob("*.png")} if depth_dir.is_dir() else set()
    conf_names = {p.stem for p in conf_dir.glob("*.png")} if conf_dir.is_dir() else set()
    names = [f"{f:06d}" for f in cap.frame_ids]
    cap.has_depth = np.array([n in depth_names for n in names])
    cap.has_confidence = np.array([n in conf_names for n in names])

    if depth_names:
        sample = cv2.imread(str(depth_dir / f"{sorted(depth_names)[0]}.png"), cv2.IMREAD_UNCHANGED)
        cap.depth_size = (int(sample.shape[1]), int(sample.shape[0]))
    if cap.video_path is not None:
        vc = cv2.VideoCapture(str(cap.video_path))
        w, h = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vc.release()
        if w and h:
            cap.rgb_size = (w, h)

    missing = int((~cap.has_depth).sum())
    if missing:
        cap.notes.append(f"{missing} of {cap.n} frames have no depth PNG (incomplete copy?); they are skipped")
    if require_depth and not cap.has_depth.any():
        raise CaptureError(
            f"{root} has no depth frames, so the LiDAR tier cannot run on it. "
            "Re-copy the export including depth/, or run the video tier: groundplan run <path> --tier video"
        )
    return cap
