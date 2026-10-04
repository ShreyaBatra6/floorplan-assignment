"""Photo-tier input: one folder per room, 2-8 stills each (HEIC or JPEG from an iPhone).

What a photo carries that the pipeline uses:
* pixels, rotated upright with the EXIF orientation;
* the focal length: ``FocalLengthIn35mmFilm`` gives the intrinsics without calibration;
* the capture time (pairs photos taken back-to-back in a doorway);
* the compass heading ``GPSImgDirection`` when Location Services were on for the Camera, which
  fixes how rooms are rotated relative to each other.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from groundplan.io.detect import IMAGE_EXT

try:  # iPhone HEIC support
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pillow_heif = None

TAGS = {v: k for k, v in ExifTags.TAGS.items()}
DIAG_35MM = float(np.hypot(36.0, 24.0))


@dataclass
class Photo:
    path: Path
    image: np.ndarray  # BGR, upright, downscaled to <= max_side
    K: np.ndarray  # intrinsics for ``image``
    focal_source: str  # exif35 | exif_mm | default
    time: dt.datetime | None
    heading_deg: float | None  # compass heading of the optical axis (degrees from north, clockwise)
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    full_size: tuple[int, int] = (0, 0)


@dataclass
class RoomPhotos:
    name: str
    folder: Path
    photos: list[Photo] = field(default_factory=list)


def _rational(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return float(x[0]) / float(x[1])
        except Exception:  # noqa: BLE001
            return None


def load_photo(path: Path, max_side: int = 1600, default_f35: float = 26.0) -> Photo:
    img = Image.open(path)
    exif = img.getexif()
    ifd = exif.get_ifd(0x8769) if exif else {}
    gps = exif.get_ifd(0x8825) if exif else {}
    img = ImageOps.exif_transpose(img).convert("RGB")
    w0, h0 = img.size
    s = min(1.0, max_side / max(w0, h0))
    if s < 1.0:
        img = img.resize((round(w0 * s), round(h0 * s)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    f35 = _rational(ifd.get(TAGS["FocalLengthIn35mmFilm"])) if ifd else None
    source = "exif35"
    if not f35:
        f35, source = default_f35, "default (no EXIF focal length)"
    fx = f35 * float(np.hypot(w, h)) / DIAG_35MM
    K = np.array([[fx, 0, (w - 1) / 2], [0, fx, (h - 1) / 2], [0, 0, 1.0]])

    t = None
    raw_t = ifd.get(TAGS["DateTimeOriginal"]) if ifd else None
    if raw_t:
        try:
            t = dt.datetime.strptime(str(raw_t), "%Y:%m:%d %H:%M:%S")
            sub = ifd.get(TAGS.get("SubsecTimeOriginal", 37521))
            if sub:
                t += dt.timedelta(seconds=float(f"0.{str(sub).strip()}"))
        except ValueError:
            t = None
    heading = None
    if gps and 17 in gps:  # GPSImgDirection
        heading = _rational(gps[17])
    return Photo(path=path, image=bgr, K=K, focal_source=source, time=t, heading_deg=heading,
                 make=str(exif.get(TAGS["Make"], "") or "") or None, model=str(exif.get(TAGS["Model"], "") or "") or None,
                 lens=str(ifd.get(TAGS.get("LensModel", 42036), "") or "") or None if ifd else None,
                 full_size=(w0, h0))


def _room_name(folder: Path) -> str:
    """'01 Living Room' -> 'Living Room' (a leading ordinal is the capture order, not the name)."""
    return re.sub(r"^\s*\d+[\s_.-]*", "", folder.name).strip() or folder.name


def load_photo_folders(root: Path, max_side: int = 1600) -> list[RoomPhotos]:
    root = Path(root)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    groups = [d for d in subdirs if any(f.suffix.lower() in IMAGE_EXT for f in d.iterdir() if f.is_file())]
    if not groups:
        groups = [root]
    rooms = []
    for g in groups:
        files = sorted(f for f in g.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT and not f.name.startswith("."))
        rp = RoomPhotos(_room_name(g), g, [load_photo(f, max_side) for f in files])
        rp.photos.sort(key=lambda p: (p.time or dt.datetime.min, p.path.name))
        rooms.append(rp)
    return rooms
