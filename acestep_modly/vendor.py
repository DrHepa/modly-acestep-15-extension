"""Install the bundled, checksum-pinned upstream source (no model weights)."""
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from .constants import ROOT, VENDOR
from .paths import safe_child


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def ensure_source(log):
    """Unpack verified regular files only; no git, shell or remote code fetch needed."""
    lock = json.loads((ROOT / "source.lock.json").read_text(encoding="utf-8"))
    archive = ROOT / "vendor" / lock["archive"]
    if sha256(archive) != lock["sha256"]:
        raise RuntimeError("Bundled upstream source checksum failed; reinstall the extension")
    VENDOR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if member.isdir():
                continue
            if not member.isfile() or member.size > 20 * 1024 * 1024:
                raise RuntimeError("Unsafe entry in bundled source archive")
            target = safe_child(VENDOR, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = bundle.extractfile(member).read()
            if target.is_file() and target.read_bytes() == data:
                continue
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                output.write(data)
                temp = Path(output.name)
            os.replace(temp, target)
    log("Pinned upstream source verified and prepared")


def sync_model_code(checkpoints, *, verify_only=False):
    """Use the same pinned model Python files as upstream, without runtime writes."""
    sources = [p for p in (VENDOR / "acestep" / "models" / "turbo").glob("*.py") if p.name != "__init__.py"]
    if not sources:
        raise RuntimeError("Pinned upstream model code is missing; run extension Repair")
    for source in sources:
        target = safe_child(checkpoints, "acestep-v15-turbo/" + source.name)
        if target.is_file() and sha256(target) == sha256(source):
            continue
        if verify_only:
            raise RuntimeError("Local ACE-Step model code is incomplete/outdated; run Repair")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            output.write(source.read_bytes())
            temp = Path(output.name)
        os.replace(temp, target)
