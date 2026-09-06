"""Create a small source-only release archive, excluding local environments and weights."""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"manifest.json", "setup.py", "processor.py", "requirements.txt", "source.lock.json", "weights.lock.json", "README.md", "VALIDATION.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".gitignore"}
SOURCE_DIRS = {"acestep_modly", "tests", "scripts", "licenses", ".github"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if str(relative) in ROOT_FILES or relative.parts[0] in SOURCE_DIRS or relative.as_posix() == "vendor/ace-step-runtime.tar.gz":
            files.append(path)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive.write(path, str(Path(manifest["id"]) / path.relative_to(ROOT)))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC validation failed")
    print(f"Packaged {len(files)} source/license files: {args.output.resolve()} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
