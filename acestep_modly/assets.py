"""Pinned file-level provisioning: verify locally, download only missing/bad files."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from .constants import ROOT
from .paths import safe_child


def read_lock():
    """Read the checked-in immutable HF file manifest."""
    return json.loads((ROOT / "weights.lock.json").read_text(encoding="utf-8"))


def valid_file(path, spec, *, full=True):
    """Check size and official SHA-256/LFS or Git blob identity."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size != spec["size"]:
        return False
    if not full:
        return True
    digest = hashlib.sha256() if "sha256" in spec else hashlib.sha1()
    if "git_blob_sha1" in spec:
        digest.update(f"blob {spec['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == spec.get("sha256", spec.get("git_blob_sha1"))


def require_assets(directory, *, full=False, lock=None):
    """Offline check used before importing the upstream runtime."""
    lock = read_lock() if lock is None else lock
    missing = [s["path"] for s in lock["files"] if not valid_file(safe_child(directory, s["path"]), s, full=full)]
    if missing:
        raise RuntimeError("ACE-Step weights are missing or incomplete: " + ", ".join(missing[:3]) + ". Run extension Repair/setup; generation never downloads weights.")


def _download(url, target, spec, log):
    partial = safe_child(target.parent, target.name + ".part")
    for attempt in range(3):
        try:
            start = partial.stat().st_size if partial.is_file() else 0
            if start >= spec["size"]:
                if valid_file(partial, spec):
                    os.replace(partial, target)
                    return
                start = 0
            headers = {"User-Agent": "Modly-ACE-Step/0.1.0"}
            if start:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if start and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise RuntimeError("Invalid resume response")
                else:
                    start = 0
                total = start
                last_report = time.monotonic()
                with partial.open("ab" if start else "wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        total += len(chunk)
                        if total > spec["size"]:
                            raise RuntimeError("Downloaded file exceeds the pinned size")
                        output.write(chunk)
                        if time.monotonic() - last_report >= 5:
                            log(f"{spec['path']}: {100 * total / spec['size']:.1f}%")
                            last_report = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
            if not valid_file(partial, spec):
                # Retry from zero after checksum failure, never accept a truncated file.
                with partial.open("wb"):
                    pass
                raise RuntimeError("Downloaded file failed its pinned checksum")
            os.replace(partial, target)
            return
        except (OSError, RuntimeError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Download failed for {spec['path']}. Run Repair to resume; existing verified files are retained.") from exc
            log(f"Retrying {spec['path']} ({attempt + 2}/3)")


def ensure_assets(directory, log, *, lock=None, downloader=_download):
    """Preserve correct weights even when extension files/venv were replaced."""
    lock = read_lock() if lock is None else lock
    directory.mkdir(parents=True, exist_ok=True)
    reused = downloaded = 0
    for index, spec in enumerate(lock["files"], 1):
        target = safe_child(directory, spec["path"])
        log(f"Checking {index}/{len(lock['files'])}: {spec['path']}")
        if valid_file(target, spec):
            reused += 1
            log("Verified locally; no download")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{lock['repo_id']}/resolve/{lock['revision']}/{quote(spec['path'], safe='/')}"
        downloader(url, target, spec, log)
        if not valid_file(target, spec):
            raise RuntimeError(f"Asset did not pass verification: {spec['path']}")
        downloaded += 1
    log(f"Weights ready: {reused} reused, {downloaded} downloaded")
    return {"reused": reused, "downloaded": downloaded}
