"""CLIP zero-shot scoring of image crops (damage verification, room types)."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from groundplan.models.registry import MODELS, cached, output_key, set_threads, torch_available

DAMAGE_PROMPTS = {
    "water_stain": ["a brown water stain on a painted wall", "a yellowish water damage stain on a ceiling",
                    "water damage discoloration on drywall"],
    "mold": ["black mold spots on a wall", "mold growth on a ceiling", "dark mildew patches on a wall"],
    "crack": ["a crack in a plaster wall", "a long hairline crack in drywall"],
    "hole": ["a hole in a drywall wall", "a broken hole punched in a wall"],
    "peeling_paint": ["peeling paint on a wall", "flaking and blistered paint"],
}
NEGATIVE_PROMPTS = [
    "a clean painted wall", "a plain white ceiling", "a framed picture on a wall", "a poster on a wall",
    "a light switch on a wall", "an electrical socket", "a soft shadow on a wall", "a wooden door", "a window",
    "a piece of furniture", "a lamp", "a curtain", "a tiled wall", "a mirror", "a cable on a wall", "a towel",
    "a kitchen cabinet", "a radiator", "a wall-mounted television", "a clock on a wall", "a shelf with objects",
    "a floor with tiles", "a carpet", "a wooden floor", "a door handle", "a cabinet handle", "a towel rail",
    "a wire basket", "a wooden wardrobe door", "a plain beige wall", "a smooth white surface", "a bathroom fitting",
]
ROOM_PROMPTS = {
    "bathroom": "a photo of a bathroom", "kitchen": "a photo of a kitchen", "bedroom": "a photo of a bedroom",
    "living room": "a photo of a living room", "hallway": "a photo of a hallway or corridor",
    "dining room": "a photo of a dining room", "office": "a photo of a home office",
    "laundry": "a photo of a laundry room", "closet": "a photo of a walk-in closet or storage room",
}


def available() -> bool:
    return torch_available()


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    set_threads()
    hub, rev = MODELS["clip"]
    model = CLIPModel.from_pretrained(hub, revision=rev).eval()
    proc = CLIPProcessor.from_pretrained(hub, revision=rev)
    with torch.no_grad():
        texts = [p for ps in DAMAGE_PROMPTS.values() for p in ps] + NEGATIVE_PROMPTS + list(ROOM_PROMPTS.values())
        tok = proc(text=texts, return_tensors="pt", padding=True)
        tf = model.get_text_features(**tok)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    index = {t: i for i, t in enumerate(texts)}
    return model, proc, tf, index


def image_features(crops: list[np.ndarray]) -> np.ndarray:
    """L2-normalised CLIP image embeddings for BGR crops (cached by crop content)."""
    if not crops:
        return np.zeros((0, 512), np.float32)
    out = []
    for crop in crops:
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        key = output_key("clip", rgb.tobytes(), "image-v1")

        def compute(rgb=rgb):
            import torch

            model, proc, _, _ = _model()
            with torch.no_grad():
                px = proc(images=[rgb], return_tensors="pt")
                f = model.get_image_features(**px)
                f = f / f.norm(dim=-1, keepdim=True)
            return {"f": f.numpy().astype(np.float32)}

        out.append(cached("clip", key, compute)["f"][0])
    return np.stack(out)


def _probs(feats: np.ndarray, texts: list[str]) -> np.ndarray:
    _, _, tf, index = _model()
    T = tf.numpy()[[index[t] for t in texts]]
    logits = 100.0 * feats @ T.T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def classify_damage(crops: list[np.ndarray]) -> list[tuple[str | None, float, float]]:
    """For each crop: (best damage class or None, its probability, best negative probability)."""
    feats = image_features(crops)
    if len(feats) == 0:
        return []
    texts = [p for ps in DAMAGE_PROMPTS.values() for p in ps] + NEGATIVE_PROMPTS
    P = _probs(feats, texts)
    out = []
    for row in P:
        k = 0
        cls_p = {}
        for cls, ps in DAMAGE_PROMPTS.items():
            cls_p[cls] = float(row[k : k + len(ps)].sum())
            k += len(ps)
        neg = float(row[k:].max())
        best = max(cls_p, key=cls_p.get)
        out.append((best, cls_p[best], neg))
    return out


def classify_room(images: list[np.ndarray]) -> tuple[str, float]:
    if not images:
        return "room", 0.0
    feats = image_features(images)
    P = _probs(feats, list(ROOM_PROMPTS.values())).mean(axis=0)
    k = int(np.argmax(P))
    return list(ROOM_PROMPTS)[k], float(P[k])
