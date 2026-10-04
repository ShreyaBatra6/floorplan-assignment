"""Video-tier input: a plain clip from the iPhone Camera app (or any phone video).

Uses the container's rotation (frames come out upright) and Apple's QuickTime metadata, which
carries the 35 mm-equivalent focal length of the recording (key
``com.apple.quicktime.camera.focal_length.35mm_equivalent``) and the device model.
Frames are sampled at a few per second, keeping the sharpest frame of each window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np

DIAG_35MM = math.hypot(36.0, 24.0)
DEFAULT_F35 = 27.0  # iPhone 1x video, stabilised (only used when the metadata is missing)


@dataclass
class VideoFrame:
    index: int
    time_s: float
    image: np.ndarray  # BGR upright, downscaled
    sharpness: float


@dataclass
class VideoClip:
    path: Path
    frames: list[VideoFrame]
    K: np.ndarray  # intrinsics for the sampled frames
    focal_source: str
    fps: float
    duration_s: float
    n_frames: int
    make: str | None = None
    model: str | None = None
    notes: list[str] = field(default_factory=list)


def _meta(container) -> dict[str, str]:
    md = dict(container.metadata)
    for s in container.streams:
        md.update({k: v for k, v in s.metadata.items() if k not in md})
    return md


def read_video(path: Path, target_fps: float = 4.0, max_frames: int = 280, max_side: int = 1280) -> VideoClip:
    path = Path(path)
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        md = _meta(c)
        fps = float(stream.average_rate or 30)
        n_total = stream.frames or 0
        duration = float(stream.duration * stream.time_base) if stream.duration else (n_total / fps if n_total else 0.0)
        est_total = n_total or int(duration * fps) or 1
        window = max(int(round(fps / target_fps)), 1)
        if est_total / window > max_frames:
            window = int(math.ceil(est_total / max_frames))
        frames: list[VideoFrame] = []
        best = None
        rotation = None
        for n, frame in enumerate(c.decode(stream)):
            if rotation is None:
                rotation = int(getattr(frame, "rotation", 0) or 0)
            small = frame.reformat(width=320, height=max(int(320 * frame.height / frame.width), 1), format="gray")
            g = small.to_ndarray()
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            if best is None or sharp > best[0]:
                best = (sharp, n, frame)
            if (n + 1) % window == 0:
                s, idx, fr = best
                img = fr.to_ndarray(format="bgr24")
                frames.append(VideoFrame(idx, idx / fps, img, s))
                best = None
        if best is not None:
            s, idx, fr = best
            frames.append(VideoFrame(idx, idx / fps, fr.to_ndarray(format="bgr24"), s))
    # upright and downscale
    k = (-(rotation or 0) // 90) % 4  # rotation is counter-clockwise degrees to display upright
    for f in frames:
        img = np.rot90(f.image, k=k).copy() if k else f.image
        h, w = img.shape[:2]
        sc = min(1.0, max_side / max(h, w))
        f.image = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else img
    h, w = frames[0].image.shape[:2] if frames else (1080, 1920)
    f35_raw = md.get("com.apple.quicktime.camera.focal_length.35mm_equivalent")
    try:
        f35 = float(f35_raw) if f35_raw else None
    except ValueError:
        f35 = None
    source = "quicktime 35mm-equivalent metadata"
    if not f35:
        f35, source = DEFAULT_F35, "default (no focal metadata; refined by structure-from-motion)"
    fx = f35 * math.hypot(w, h) / DIAG_35MM
    K = np.array([[fx, 0, (w - 1) / 2], [0, fx, (h - 1) / 2], [0, 0, 1.0]])
    clip = VideoClip(path, frames, K, source, fps, duration, n_total or len(frames),
                     md.get("com.apple.quicktime.make"), md.get("com.apple.quicktime.model"))
    if not frames:
        clip.notes.append("no frames could be decoded")
    return clip
