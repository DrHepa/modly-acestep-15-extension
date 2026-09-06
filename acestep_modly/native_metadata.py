"""Narrow repair for the published cuSPARSELt 0.8.0 SBSA internal wheel tag.

The official aarch64 wheel contains an internal ``manylinux2014_sbsa`` tag.
Pip installs that wheel by filename but rejects the internal tag at pip check.
This runs in the extension venv; it does not waive any pip/native failure.
"""
import argparse
import base64
import csv
import ctypes
from email.parser import BytesParser
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import sys
import tempfile


NAME = "nvidia-cusparselt-cu13"
VERSION = "0.8.0"
INFO = "nvidia_cusparselt_cu13-0.8.0.dist-info"
OLD_TAG = "py3-none-manylinux2014_sbsa"
NEW_TAG = "py3-none-manylinux2014_aarch64"
LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"


def _record_fields(data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return ["sha256=" + digest, str(len(data))]


def _atomic_write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def repair_cusparselt_tag(flavor):
    """Validate native ARM64 compatibility before changing only WHEEL/RECORD."""
    if (sys.platform, platform.machine().lower(), flavor) != ("linux", "aarch64", "cu130"):
        return False
    dist = importlib.metadata.distribution(NAME)
    if dist.metadata.get_all("Name") != [NAME] or dist.metadata.get_all("Version") != [VERSION]:
        raise RuntimeError("cuSPARSELt metadata repair requires the exact nvidia-cusparselt-cu13 0.8.0 distribution")
    wheel_key, record_key = INFO + "/WHEEL", INFO + "/RECORD"
    files = {str(file) for file in (dist.files or [])}
    if not {wheel_key, record_key, LIBRARY}.issubset(files):
        raise RuntimeError("cuSPARSELt WHEEL, RECORD or native library is missing from the distribution")
    wheel, record, library = (Path(dist.locate_file(key)) for key in (wheel_key, record_key, LIBRARY))
    original = wheel.read_bytes()
    tags = BytesParser().parsebytes(original).get_all("Tag", [])
    if tags not in ([OLD_TAG], [NEW_TAG]):
        raise RuntimeError("Unknown cuSPARSELt WHEEL tags; refusing a generic platform override")
    if original.count(("Tag: " + tags[0]).encode("ascii")) != 1:
        raise RuntimeError("Unexpected cuSPARSELt WHEEL formatting")
    corrected = original.replace(("Tag: " + OLD_TAG).encode(), ("Tag: " + NEW_TAG).encode())
    uncorrected = corrected.replace(("Tag: " + NEW_TAG).encode(), ("Tag: " + OLD_TAG).encode())
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    if any(len(row) != 3 for row in rows) or len({row[0] for row in rows}) != len(rows):
        raise RuntimeError("Malformed or duplicate cuSPARSELt RECORD entries")
    wheel_row = next(row for row in rows if row[0] == wheel_key)
    record_row = next(row for row in rows if row[0] == record_key)
    # Recognize only an exact interrupted prior repair, never arbitrary hash drift.
    if wheel_row[1:] not in (_record_fields(original), _record_fields(uncorrected)) or record_row[1:] != ["", ""]:
        raise RuntimeError("cuSPARSELt WHEEL/RECORD integrity mismatch; reinstall this dependency before Repair")
    with library.open("rb") as file:
        header = file.read(64)
    if len(header) != 64 or header[:7] != b"\x7fELF\x02\x01\x01" or header[16:20] != b"\x03\x00\xb7\x00":
        raise RuntimeError("cuSPARSELt native library must be an ELF64 little-endian AArch64 shared object")
    ctypes.CDLL(str(library))  # An unresolved dependency/symbol is a fatal error.
    changed = original != corrected or wheel_row[1:] != _record_fields(corrected)
    if changed:
        wheel_row[1:] = _record_fields(corrected)
        output = io.StringIO(newline="")
        csv.writer(output).writerows(rows)
        if original != corrected:
            _atomic_write(wheel, corrected)
        # If interrupted here, the exact original RECORD is recognized above.
        _atomic_write(record, output.getvalue().encode("utf-8"))
        print("[ACE-Step setup] Validated cuSPARSELt 0.8.0 ARM64 native load; normalized its SBSA WHEEL tag and RECORD", flush=True)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", required=True)
    repair_cusparselt_tag(parser.parse_args().flavor)
