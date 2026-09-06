"""Resolve actual host storage without relying on a user-specific path."""
import json
import os
import stat
from pathlib import Path

from .constants import EXTENSION_ID, ROOT, STATE_FILE


def absolute_dir(value, label):
    """Validate a native absolute directory without creating it."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an absolute directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} is not a directory")
    return path


def _explicit(values, keys):
    paths = {absolute_dir(values[k], k) for k in keys if values.get(k)}
    if len(paths) > 1:
        raise ValueError("Conflicting models directories; specify only one storage root")
    return next(iter(paths), None)


def _settings_candidates(root, payload, env):
    candidates = [root.parent.parent / "settings.json"]
    if env.get("MODLY_USER_DATA"):
        candidates.insert(0, Path(env["MODLY_USER_DATA"]) / "settings.json")
    # The setup interpreter is installed below Modly's userData directory.
    if payload.get("python_exe"):
        candidates.extend(p / "settings.json" for p in Path(payload["python_exe"]).parents)
    if os.name == "nt":
        config = Path(env.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        config = Path(env.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    candidates.extend(config / name / "settings.json" for name in ("modly", "Modly"))
    return list(dict.fromkeys(candidates))


def resolve_models_root(payload=None, *, root=ROOT, env=None, state_file=STATE_FILE):
    """Prefer explicit context/env, then settings bound to this installed extension."""
    payload = payload or {}
    env = os.environ if env is None else env
    explicit = _explicit(payload, ("models_dir", "modelsDir"))
    override = _explicit(env, ("MODLY_MODELS_DIR", "MODELS_DIR"))
    if explicit is not None:
        return _disjoint(explicit, root)
    if override is not None:
        return _disjoint(override, root)
    matches = set()
    for candidate in _settings_candidates(root, payload, env):
        if not candidate.is_file():
            continue
        try:
            settings = json.loads(candidate.read_text(encoding="utf-8"))
            extensions = absolute_dir(settings.get("extensionsDir", str(candidate.parent / "extensions")), "extensionsDir")
            installed = extensions / EXTENSION_ID
            if installed.resolve() != root.resolve():
                continue
            matches.add(absolute_dir(settings.get("modelsDir", str(candidate.parent / "models")), "modelsDir"))
        except (ValueError, OSError, AttributeError) as exc:
            if candidate == root.parent.parent / "settings.json":
                raise ValueError("Cannot read Modly settings; repair settings or set MODELS_DIR") from exc
    if len(matches) > 1:
        raise ValueError("Multiple Modly installations claim this extension with different storage roots")
    if matches:
        return _disjoint(matches.pop(), root)
    # Records support explicit/manual setup and are recreated on every successful setup.
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("extension_root") == str(root.resolve()):
            return _disjoint(absolute_dir(state["models_root"], "saved models root"), root)
    # An arbitrary directory named "extensions" does not prove Modly's storage root.
    # Fail closed instead of downloading a large snapshot into a guessed location.
    raise ValueError("Cannot resolve Modly models_dir. Set MODELS_DIR to the path shown in Modly Settings > Storage and run setup/Repair.")


def _disjoint(models_root, root):
    models_root, root = models_root.resolve(), root.resolve()
    if models_root == root or root in models_root.parents or models_root in root.parents:
        raise ValueError("models_dir must be separate from the extension/runtime directory")
    return models_root


def safe_child(root, relative):
    """Reject escapes, symlinks and Windows reparse points beneath an owned root."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Invalid relative asset path")
    parts = relative.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError("Unsafe relative asset path")
    current = root
    for part in parts:
        current = current / part
        attributes = getattr(current.lstat(), "st_file_attributes", 0) if current.exists() or current.is_symlink() else 0
        if current.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("Owned asset paths must not contain links or junctions")
    if root.resolve() not in current.resolve().parents:
        raise ValueError("Asset path escapes its storage root")
    return current


def checkpoint_dir(models_root):
    """Stable path outside the replaceable extension directory."""
    return safe_child(models_root, EXTENSION_ID + "/checkpoints")
