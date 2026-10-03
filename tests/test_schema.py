import json
from pathlib import Path

from groundplan.contract import json_schema

SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "groundplan.schema.json"


def test_published_schema_is_current():
    """The committed schema must match the contract models (run `groundplan schema` to refresh)."""
    assert SCHEMA.exists(), "schema/groundplan.schema.json missing"
    assert json.loads(SCHEMA.read_text(encoding="utf-8")) == json_schema()
