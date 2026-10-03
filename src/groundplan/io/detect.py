"""Decide which tier a capture is, from what is on disk."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from groundplan.io.stray import CaptureError, find_stray_root

IMAGE_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png"}
VIDEO_EXT = {".mov", ".mp4", ".m4v"}


@dataclass
class Detected:
    tier: str
    root: Path
    detail: str


def unpack_if_zip(path: Path, cache_dir: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".zip":
        dest = cache_dir / "unzipped" / path.stem
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest)
        return dest
    return path


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT
                  and not p.name.startswith("."))


def _videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT
                  and not p.name.startswith("."))


def detect(path: Path) -> Detected:
    path = Path(path)
    if not path.exists():
        raise CaptureError(f"{path} does not exist")
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXT:
            return Detected("video", path, f"video file {path.name}")
        raise CaptureError(f"{path.name}: expected a Stray Scanner folder, a video, or a folder of room photo folders")

    stray = find_stray_root(path)
    if stray is not None:
        depth = stray / "depth"
        if depth.is_dir() and any(depth.glob("*.png")):
            return Detected("lidar", stray, "Stray Scanner export with depth")
        if (stray / "rgb.mp4").exists():
            return Detected("video", stray / "rgb.mp4",
                            "Stray Scanner export without depth: running the video tier on rgb.mp4")
        raise CaptureError(f"{stray} is a Stray Scanner export with neither depth nor video")

    subdirs = sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))
    room_dirs = [d for d in subdirs if _images(d)]
    if room_dirs:
        return Detected("photo", path, f"{len(room_dirs)} room photo folder(s)")
    if _images(path):
        return Detected("photo", path, "one folder of photos (treated as a single room)")
    vids = _videos(path)
    if len(vids) >= 1:
        return Detected("video", vids[0] if len(vids) == 1 else path, f"{len(vids)} video file(s)")
    raise CaptureError(f"{path}: no Stray Scanner export, video, or photos found")
