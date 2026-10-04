"""Video-tier input: a plain clip from the iPhone Camera app (or any phone video).

Uses the container's rotation (frames come out upright) and Apple's QuickTime metadata, which
carries the 35 mm-equivalent focal length of the recording (key
``com.apple.quicktime.camera.focal_length.35mm_equivalent``) and the device model.
Frames are sampled at a few per second, keeping the sharpest frame of each window, and more often
where the view changes fast: every decoded frame is tracked (optical flow at 320 px) from the last
kept frame, and a frame is kept early once the view has moved by about an eighth of its width or
lost much of what it tracked. Fast turns are where structure from motion breaks into pieces, and
denser frames there keep consecutive frames overlapping. Over the frame cap, the frames that moved
least relative to their predecessor are thinned first.
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
    motion: float = 0.0  # view change since the previous kept frame (fraction of the image width)


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


class _ViewChange:
    """How far the view has moved since the last kept frame (tracked corners, global shift fallback)."""

    def __init__(self, min_points: int = 30):
        self.min_points = min_points
        self.key = self.prev = self.p0 = self.p = None

    def reset(self, g: np.ndarray) -> None:
        self.key = self.prev = g
        pts = cv2.goodFeaturesToTrack(g, maxCorners=200, qualityLevel=0.005, minDistance=8)
        self.p0 = self.p = pts.reshape(-1, 2) if pts is not None else np.zeros((0, 2), np.float32)

    def update(self, g: np.ndarray) -> tuple[float, float]:
        """(motion as a fraction of the width, fraction of the key frame's points still tracked)."""
        w = g.shape[1]
        if len(self.p0) >= self.min_points and len(self.p):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.p.reshape(-1, 1, 2).astype(np.float32), None,
                                                  winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            self.p, self.p0 = nxt.reshape(-1, 2)[ok], self.p0[ok]
            self.prev = g
            kept = len(self.p) / max(len(self.p0) + (~ok).sum(), 1)
            if len(self.p) >= 8:
                return float(np.median(np.linalg.norm(self.p - self.p0, axis=1))) / w, kept
            return 1.0, 0.0
        (dx, dy), resp = cv2.phaseCorrelate(self.key.astype(np.float32), g.astype(np.float32))
        self.prev = g
        return (math.hypot(dx, dy) / w, 1.0) if resp > 0.05 else (1.0, 0.0)


def _thin(frames: list[VideoFrame], max_frames: int) -> list[VideoFrame]:
    """Drop the frames that add least (smallest motion from their predecessor) until under the cap."""
    frames = list(frames)
    while len(frames) > max_frames:
        k = 1 + int(np.argmin([f.motion for f in frames[1:-1]]))
        if k + 1 < len(frames):
            frames[k + 1].motion += frames[k].motion
        frames.pop(k)
    return frames


def read_video(path: Path, target_fps: float = 4.0, max_frames: int = 280, max_side: int = 1280,
               max_motion: float = 0.12, min_tracked: float = 0.6) -> VideoClip:
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
        tracker = _ViewChange()
        since = 0
        motion, tracked = 0.0, 1.0
        early = 0
        for n, frame in enumerate(c.decode(stream)):
            if rotation is None:
                rotation = int(getattr(frame, "rotation", 0) or 0)
            small = frame.reformat(width=320, height=max(int(320 * frame.height / frame.width), 1), format="gray")
            g = small.to_ndarray()
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            if tracker.key is None:
                tracker.reset(g)
            else:
                motion, tracked = tracker.update(g)
            if best is None or sharp > best[0]:
                best = (sharp, n, frame, motion)
            since += 1
            fast = tracker.key is not None and since >= 2 and (motion > max_motion or tracked < min_tracked)
            if since >= window or fast:
                s, idx, fr, mo = best if not fast else (sharp, n, frame, motion)  # on a fast turn keep the newest
                frames.append(VideoFrame(idx, idx / fps, fr.to_ndarray(format="bgr24"), s, mo))
                early += fast and since < window
                best, since, motion, tracked = None, 0, 0.0, 1.0
                tracker.reset(g)
        if best is not None:
            s, idx, fr, mo = best
            frames.append(VideoFrame(idx, idx / fps, fr.to_ndarray(format="bgr24"), s, mo))
        if len(frames) > max_frames:
            frames = _thin(frames, max_frames)
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
    if early:
        clip.notes.append(f"{early} extra frame(s) kept where the view changed fast (turns)")
    return clip
