"""Venv-side, exclusively locked setup provisioning of persistent model assets."""
import argparse
from filelock import FileLock

from .assets import ensure_assets
from .constants import EXTENSION_ID
from .paths import absolute_dir, checkpoint_dir, safe_child
from .vendor import sync_model_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True)
    args = parser.parse_args()
    models = absolute_dir(args.models_root, "models root")
    models.mkdir(parents=True, exist_ok=True)
    owned = safe_child(models, EXTENSION_ID)
    owned.mkdir(parents=True, exist_ok=True)
    lock_path = safe_child(owned, "assets.lock")
    log = lambda message: print("[ACE-Step setup] " + message, flush=True)
    log("Waiting for the model asset lock (another setup/generation may be active)")
    with FileLock(str(lock_path), timeout=3600):
        checkpoints = checkpoint_dir(models)
        ensure_assets(checkpoints, log)
        sync_model_code(checkpoints)


if __name__ == "__main__":
    main()
