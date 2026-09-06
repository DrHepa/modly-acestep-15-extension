"""One-shot text-to-audio process entry for the current upstream Modly protocol."""
import json
import os
import signal
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from acestep_modly.assets import require_assets
from acestep_modly.constants import EXTENSION_ID, NODE_ID, UPSTREAM_REVISION, VENDOR
from acestep_modly.offline import configure, deny_network
from acestep_modly.parameters import normalize
from acestep_modly.paths import absolute_dir, checkpoint_dir, resolve_models_root, safe_child
from acestep_modly.protocol import Protocol, configure_stdio
from acestep_modly.vendor import sync_model_code


def process(payload, protocol):
    if not isinstance(payload, dict):
        raise ValueError("Process payload must be an object")
    node_id = payload.get("nodeId") or (payload.get("input") or {}).get("nodeId")
    if node_id != NODE_ID:
        raise ValueError(f"Unsupported nodeId: {node_id}")
    values = normalize(payload)
    workspace = absolute_dir(payload.get("workspaceDir"), "workspaceDir")
    temp_root = absolute_dir(payload.get("tempDir"), "tempDir")
    if not (VENDOR / "acestep" / "handler.py").is_file():
        raise RuntimeError("ACE-Step runtime is not prepared. Run extension setup/Repair first.")
    models = resolve_models_root(payload)
    checkpoints = checkpoint_dir(models)
    require_assets(checkpoints)
    sync_model_code(checkpoints, verify_only=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock, Timeout
    lock_path = safe_child(models, EXTENSION_ID + "/assets.lock")
    try:
        with FileLock(str(lock_path), timeout=0):
            require_assets(checkpoints)
            run_id = uuid.uuid4().hex
            output_dir = safe_child(workspace, "Workflows/ACE-Step-1.5/" + run_id)
            output_dir.mkdir(parents=True, exist_ok=False)
            with tempfile.TemporaryDirectory(prefix="modly-acestep-", dir=temp_root) as temp:
                configure(checkpoints, Path(temp))
                deny_network()
                from acestep_modly.runtime import generate
                output, metadata = generate(values, checkpoints, output_dir, Path(temp), protocol.progress, protocol.log)
                metadata.update({"parameters": values, "upstream_revision": UPSTREAM_REVISION})
                output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"filePath": str(output)}
    except Timeout as exc:
        raise RuntimeError("ACE-Step is busy with another generation or setup. Wait for it to finish and retry.") from exc


def interrupted(*_):
    raise KeyboardInterrupt


def main():
    configure_stdio()
    protocol = Protocol()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raw = sys.stdin.buffer.readline(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("Process request exceeds 1 MiB")
        payload = json.loads(raw.decode("utf-8"))
        with protocol.capture_logs(), protocol.heartbeat():
            try:
                result = process(payload, protocol)
            except BaseException:
                traceback.print_exc()
                raise
        protocol.progress(100, "Audio ready")
        protocol.emit({"type": "done", "result": result})
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        message = "ACE-Step interrupted" if isinstance(exc, KeyboardInterrupt) else f"{type(exc).__name__}: {exc}"
        protocol.log(message)
        protocol.emit({"type": "error", "message": message[:8000]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
