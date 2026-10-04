"""Staged damage for testing: procedural decals projected onto a real surface in every view.

Used to measure detection and extent accuracy on real footage with known ground truth (the decal's
exact metric extent), complementing physically staged damage in the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from groundplan.damage.ortho import SurfaceGeom, View

TEX_RES = 0.002  # metres per texture pixel


@dataclass
class Decal:
    kind: str  # water_stain | crack
    u0: float
    v0: float
    width: float
    height: float
    seed: int = 0

    def texture(self) -> np.ndarray:
        """RGBA texture (premultiplied-free), alpha 0 outside the damage."""
        w = max(int(self.width / TEX_RES), 8)
        h = max(int(self.height / TEX_RES), 8)
        rng = np.random.default_rng(self.seed)
        if self.kind == "water_stain":
            yy, xx = np.mgrid[0:h, 0:w]
            nx, ny = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
            ang = np.arctan2(ny, nx)
            wobble = 1 + 0.12 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.08 * np.sin(5 * ang + rng.uniform(0, 6))
            r = np.sqrt(nx**2 + ny**2) / wobble
            inside = r < 0.95
            ring = np.clip(1 - np.abs(r - 0.88) / 0.08, 0, 1)
            body = np.clip(1 - r, 0, 1) ** 0.5
            noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
            alpha = np.clip(0.35 * body + 0.55 * ring + 0.08 * noise, 0, 0.85) * inside
            col = np.zeros((h, w, 3), np.float32)
            col[...] = (60, 110, 150)  # BGR yellow-brown
            col -= (ring * 35)[..., None]
            return np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8)
        # crack: a wandering dark polyline across the decal's diagonal
        tex = np.zeros((h, w, 4), np.uint8)
        n = 40
        t = np.linspace(0, 1, n)
        xs = t * (w - 1)
        ys = t * (h - 1) + np.cumsum(rng.normal(0, h * 0.02, n))
        ys = np.clip(ys - ys.mean() + (h - 1) / 2, 2, h - 3)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(tex, [pts], False, (45, 45, 50, 255), thickness=max(int(0.004 / TEX_RES), 1),
                      lineType=cv2.LINE_AA)
        return tex


def apply_decals(views: list[View], surface: SurfaceGeom, decals: list[Decal]) -> list[View]:
    """Return copies of the views with each decal painted where the surface is visible."""
    out = []
    for view in views:
        img = view.image.copy()
        for d in decals:
            tex = d.texture()
            th, tw = tex.shape[:2]
            corners_uv = [(d.u0, d.v0 + d.height), (d.u0 + d.width, d.v0 + d.height), (d.u0 + d.width, d.v0),
                          (d.u0, d.v0)]
            pts = np.array([surface.origin + (u - surface.u0) * surface.u_axis + (v - surface.v0) * surface.v_axis
                            for u, v in corners_uv])
            R, t = view.T_wc[:3, :3], view.T_wc[:3, 3]
            cam = (pts - t) @ R
            if np.any(cam[:, 2] < 0.2):
                continue
            if (view.center - pts.mean(0)) @ surface.normal <= 0:
                continue
            u = view.K[0, 0] * cam[:, 0] / cam[:, 2] + view.K[0, 2]
            v = view.K[1, 1] * cam[:, 1] / cam[:, 2] + view.K[1, 2]
            dst = np.stack([u, v], 1).astype(np.float32)
            src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(tex, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
            a = warped[..., 3:4].astype(np.float32) / 255.0
            if view.depth is not None and view.K_depth is not None:
                # do not paint over things standing in front of the wall
                zc = cam[:, 2].mean()
                dmap = cv2.resize(view.depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                a = a * ((dmap <= 0) | (dmap > zc - 0.1))[..., None]
            img = (img * (1 - a) + warped[..., :3] * a).astype(np.uint8)
        out.append(View(img, view.K, view.T_wc, view.depth, view.K_depth, view.key))
    return out
