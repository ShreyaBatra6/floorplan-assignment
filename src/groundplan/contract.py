"""The output contract.

Every capture, at every tier, produces exactly one ``Plan`` document. The JSON Schema in
``schema/groundplan.schema.json`` is generated from these models (``groundplan schema``) and every
run validates its own output against it before writing.

Conventions
-----------
* Plan frame: right-handed, metres. ``x`` to the right and ``y`` up on the rendered plan; heights
  are measured from the room's floor plane. Polygons are counter-clockwise and trace the
  *interior* face of the walls.
* Surface coordinates (damage regions): for a wall, ``u`` runs along the wall from its start corner
  and ``v`` is height above the floor; for floor and ceiling, ``(u, v)`` are plan ``(x, y)``.
* Every number with physical meaning is a ``Measurement``: a value plus a two-sided interval at
  ``confidence``. ``calibrated`` is false while the tier's interval multipliers are still priors
  rather than fitted on the benchmark.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["photo", "video", "lidar"]
Unit = Literal["m", "m2", "deg", "ea"]
DamageClass = Literal["water_stain", "mold", "crack", "hole", "peeling_paint", "other"]
OpeningKind = Literal["door", "window", "open_passage"]
SurfaceKind = Literal["wall", "floor", "ceiling"]
Severity = Literal["low", "medium", "high"]

Point2 = tuple[float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Measurement(_Model):
    """A measured quantity with a calibrated two-sided interval."""

    value: float
    lo: float
    hi: float
    unit: Unit
    confidence: float = Field(0.9, gt=0.0, lt=1.0)
    sigma_abs: float = Field(ge=0.0, description="additive 1-sigma error component, in `unit`")
    sigma_log: float = Field(ge=0.0, description="multiplicative 1-sigma error component (log scale)")
    method: str = Field(description="how the value was obtained")
    calibrated: bool = Field(description="false while interval multipliers are priors, not benchmark fits")


class DeviceInfo(_Model):
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    focal_length_35mm: float | None = None
    software: str | None = None


class CaptureQuality(_Model):
    frames_low_light: float = Field(0.0, ge=0.0, le=1.0, description="fraction of used frames judged too dark")
    frames_blurred: float = Field(0.0, ge=0.0, le=1.0, description="fraction of frames rejected as blurred")
    mirror_suspected: list[str] = Field(default_factory=list, description="surface ids treated as mirrors")
    specular_floor_suspected: bool = False
    notes: list[str] = Field(default_factory=list)


class Capture(_Model):
    id: str
    tier: Tier
    source: str = Field(description="path the pipeline was run on")
    app: str | None = Field(None, description="capture application, e.g. 'Stray Scanner' or 'iPhone Camera'")
    device: DeviceInfo | None = None
    started_at: str | None = None
    duration_s: float | None = None
    frames_total: int | None = None
    frames_used: int | None = None
    photos_per_room: dict[str, int] | None = None
    quality: CaptureQuality = Field(default_factory=CaptureQuality)


class ScaleCue(_Model):
    name: str
    log_scale: float = Field(description="log of the multiplicative correction this cue suggests")
    sigma_log: float = Field(ge=0.0)
    used: bool
    detail: str = ""


class ScaleInfo(_Model):
    source: str = Field(description="what fixes metric scale: 'lidar', 'metric_depth', 'fused', ...")
    sigma_log: float = Field(ge=0.0, description="residual 1-sigma log-scale uncertainty after fusion")
    cues: list[ScaleCue] = Field(default_factory=list)


class CalibrationInfo(_Model):
    version: str
    fitted: bool
    source: str = Field(description="benchmark run the multipliers were fitted on, or 'prior'")


class Wall(_Model):
    id: str
    room_id: str
    index: int
    surface_id: str
    start: Point2
    end: Point2
    length: Measurement
    height: Measurement
    outward_normal_deg: float = Field(description="plan-frame azimuth of the outward normal, degrees")
    adjacent_room_id: str | None = None
    thickness: Measurement | None = None
    observed_fraction: float = Field(ge=0.0, le=1.0)


class Opening(_Model):
    id: str
    room_id: str
    wall_id: str
    kind: OpeningKind
    offset: Measurement = Field(description="distance from the wall's start corner to the opening's near jamb")
    width: Measurement
    height: Measurement
    sill_height: Measurement | None = None
    connects_to_room_id: str | None = None
    detection_confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class CeilingLevel(_Model):
    height: Measurement
    area_fraction: float = Field(ge=0.0, le=1.0)


class Surface(_Model):
    id: str
    room_id: str
    kind: SurfaceKind
    wall_id: str | None = None
    area: Measurement


class RoomType(_Model):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


class RoomCoverage(_Model):
    floor_observed_fraction: float = Field(ge=0.0, le=1.0)
    walls_observed_fraction: float = Field(ge=0.0, le=1.0)
    ceiling_observed: bool


class Room(_Model):
    id: str
    name: str
    type: RoomType
    is_connector: bool = Field(False, description="hallway / corridor joining other rooms")
    source: str | None = Field(None, description="photo folder or capture segment this room came from")
    polygon: list[Point2] = Field(min_length=3)
    floor_area: Measurement
    perimeter: Measurement
    ceiling_height: Measurement
    ceiling_levels: list[CeilingLevel] = Field(default_factory=list)
    walls: list[Wall]
    openings: list[Opening] = Field(default_factory=list)
    surfaces: list[Surface]
    coverage: RoomCoverage


class Adjacency(_Model):
    room_a: str
    room_b: str
    kind: Literal["door", "open_passage", "shared_wall"]
    via_opening_ids: list[str] = Field(default_factory=list)
    shared_wall_length: Measurement | None = None
    evidence: str = ""


class Overlap(_Model):
    room_a: str
    room_b: str
    area_m2: float


class DriftReport(_Model):
    enabled: bool
    method: str
    submaps: int = 0
    loop_closures: int = 0
    max_correction_m: float = 0.0
    mean_correction_m: float = 0.0
    notes: str = ""


class StitchedPlan(_Model):
    frame: str
    room_ids: list[str]
    adjacency: list[Adjacency]
    footprint_area: Measurement = Field(description="sum of room interior floor areas (no overlaps)")
    bounding_box_width: Measurement
    bounding_box_depth: Measurement
    overlaps: list[Overlap] = Field(default_factory=list)
    stitch_method: str
    drift: DriftReport


class DamageRegion(_Model):
    id: str
    room_id: str
    surface_id: str
    damage_class: DamageClass
    confidence: float = Field(ge=0.0, le=1.0)
    polygon_uv: list[Point2] = Field(min_length=3)
    area: Measurement
    width: Measurement
    height: Measurement
    center_height_above_floor: Measurement | None = None
    views: int = Field(ge=1)
    evidence_image: str | None = None


class ConcealedFlag(_Model):
    id: str
    rule_id: str
    rule: str = Field(description="the rule that fired, in words")
    condition: str = Field(description="the evaluated condition, with the values that satisfied it")
    suspected: str
    room_id: str
    surface_ids: list[str]
    region_ids: list[str]
    severity: Severity
    recommended_verification: str
    reference: str | None = None


class ScopeItem(_Model):
    id: str
    code: str
    description: str
    room_id: str
    surface_id: str
    region_ids: list[str] = Field(default_factory=list)
    flag_ids: list[str] = Field(default_factory=list)
    quantity: Measurement
    unit_label: Literal["SF", "LF", "EA"]
    quantity_imperial: float
    derivation: str


class Runtime(_Model):
    groundplan_version: str
    git_commit: str | None = None
    started_at: str
    total_s: float
    stages: dict[str, float]
    machine: str
    model_cache: Literal["hit", "miss", "mixed", "off", "n/a"] = "n/a"


class Plan(_Model):
    schema_version: str
    capture: Capture
    units: dict[str, str] = Field(default_factory=lambda: {"length": "m", "area": "m2", "angle": "deg"})
    confidence_level: float = Field(0.9, gt=0.0, lt=1.0)
    calibration: CalibrationInfo
    scale: ScaleInfo
    rooms: list[Room]
    stitched: StitchedPlan
    damage_regions: list[DamageRegion] = Field(default_factory=list)
    concealed_damage_flags: list[ConcealedFlag] = Field(default_factory=list)
    scope: list[ScopeItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    runtime: Runtime


def json_schema() -> dict:
    schema = Plan.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/groundplan/groundplan/schema/groundplan.schema.json"
    schema["title"] = "groundplan Plan"
    return schema
