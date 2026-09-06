"""Validate workflow inputs and keep runtime defaults identical to the manifest."""
import json
import math

from .constants import NODE_ID, ROOT


def normalize(payload):
    if not isinstance(payload, dict):
        raise ValueError("Process payload must be a JSON object")
    input_data, params = payload.get("input") or {}, payload.get("params") or {}
    if not isinstance(input_data, dict) or not isinstance(params, dict):
        raise ValueError("input and params must be objects")
    node_id = payload.get("nodeId") or input_data.get("nodeId")
    if node_id != NODE_ID or (input_data.get("nodeId") and input_data["nodeId"] != node_id):
        raise ValueError(f"Unsupported/conflicting nodeId: expected {NODE_ID}")
    schema = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["nodes"][0]["params_schema"]
    values = {}
    for item in schema:
        value = params.get(item["id"], item["default"])
        if item["type"] in ("int", "float"):
            if isinstance(value, bool):
                raise ValueError(f"{item['id']} must be numeric, not boolean")
            number = float(value)
            if not math.isfinite(number) or not item["min"] <= number <= item["max"]:
                raise ValueError(f"{item['id']} must be between {item['min']} and {item['max']}")
            if item["type"] == "int" and not number.is_integer():
                raise ValueError(f"{item['id']} must be an integer")
            value = int(number) if item["type"] == "int" else number
        elif item["type"] == "select":
            if isinstance(value, bool):
                value = str(value).lower()
            if value not in [option["value"] for option in item["options"]]:
                raise ValueError(f"Invalid choice for {item['id']}")
        elif not isinstance(value, str):
            raise ValueError(f"{item['id']} must be text")
        values[item["id"]] = value
    text = input_data.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Connected input.text must be text")
    if input_data.get("filePath") and not text.strip():
        raise ValueError("Connect a Text node, not an audio/mesh file")
    values["caption"] = text.strip() or values["caption"].strip()
    if not values["caption"] or len(values["caption"]) > 4096:
        raise ValueError("Provide a music description of 1-4096 characters")
    values["instrumental"] = values["instrumental"] == "true"
    values["thinking"] = values["thinking"] == "true"
    values["lyrics"] = "[Instrumental]" if values["instrumental"] else values["lyrics"].replace("\\n", "\n").strip()
    if len(values["lyrics"]) > 16384:
        raise ValueError("Lyrics must be at most 16384 characters")
    if 0 < values["bpm"] < 30:
        raise ValueError("BPM must be 0 (automatic) or 30-300")
    return values
