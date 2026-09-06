"""Modly process Install/Repair: isolated Python, then persistent verified weights."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from acestep_modly.constants import STATE_FILE, UPSTREAM_REVISION
from acestep_modly.paths import absolute_dir, resolve_models_root
from acestep_modly.platforms import probe_cuda_version, select_lane, torch_requirements
from acestep_modly.protocol import configure_stdio
from acestep_modly.vendor import ensure_source


def log(message):
    print("[ACE-Step setup] " + message, flush=True)


def parse_args(argv):
    """Accept one upstream JSON argument or legacy positional setup arguments."""
    if len(argv) == 2:
        context = json.loads(argv[1])
        if not isinstance(context, dict):
            raise ValueError("Setup JSON must be an object")
        return context
    if len(argv) >= 4 and len(argv) <= 5:
        return {"python_exe": argv[1], "ext_dir": argv[2], "gpu_sm": int(argv[3]), "cuda_version": int(argv[4]) if len(argv) == 5 else 0}
    raise ValueError("Expected one Modly JSON argument or <python_exe> <ext_dir> <gpu_sm> [cuda_version]")


def run(command, stage, *, env=None):
    log(stage)
    subprocess.run([str(x) for x in command], check=True, cwd=ROOT, env=env)


def venv_python(root):
    return root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv(bootstrap):
    """Keep a compatible venv; preserve an incompatible one as a recoverable backup."""
    python = venv_python(ROOT)
    if python.is_file():
        try:
            result = subprocess.run([str(python), "-c", "import sys; print('%s.%s' % sys.version_info[:2])"], check=True, capture_output=True, text=True, timeout=30)
            if result.stdout.strip() != f"{sys.version_info.major}.{sys.version_info.minor}":
                raise ValueError("Python ABI changed")
            return python
        except (subprocess.SubprocessError, OSError, ValueError):
            backup = ROOT / ("venv-incompatible-" + uuid.uuid4().hex[:8])
            (ROOT / "venv").rename(backup)
            log(f"Previous incompatible runtime preserved as {backup.name}")
    run([bootstrap, "-m", "venv", ROOT / "venv"], "Creating extension venv")
    return python


def setup(context):
    ext = absolute_dir(context.get("ext_dir", str(ROOT)), "ext_dir")
    if ext != ROOT:
        raise ValueError("ext_dir must identify this extension, not another directory")
    bootstrap = Path(context.get("python_exe", sys.executable))
    if not bootstrap.is_absolute() or not bootstrap.is_file():
        raise ValueError("python_exe must identify an existing absolute Python path")
    identity = subprocess.run([str(bootstrap), "-c", "import sys; print('%s.%s' % sys.version_info[:2])"], check=True, capture_output=True, text=True, timeout=30)
    if identity.stdout.strip() != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError("Run setup.py with the same Python interpreter supplied in python_exe")
    lane = select_lane(context, driver_cuda=probe_cuda_version())
    models = resolve_models_root(context)
    log(f"Storage: {models}; lane: {lane['system']}/{lane['arch']} Python {lane['python']} {lane['flavor']}")
    ensure_source(log)
    python = ensure_venv(bootstrap)
    run([python, "-m", "pip", "install", "pip==25.3", "setuptools==71.1.0", "wheel==0.45.1"], "Preparing pip tooling")
    native = torch_requirements(lane)
    run([python, "-m", "pip", "install", *native, "--index-url", lane["index"]], "Installing pinned native PyTorch lane")
    # Constrain native versions so an auxiliary dependency cannot replace CUDA/Python ABI.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as file:
        file.write("\n".join(native) + "\n")
        constraints = Path(file.name)
    try:
        run([python, "-m", "pip", "install", "-r", ROOT / "requirements.txt", "-c", constraints], "Installing inference dependencies")
    finally:
        constraints.unlink(missing_ok=True)
    run([python, "-m", "acestep_modly.native_metadata", "--flavor", lane["flavor"]], "Validating known native wheel metadata")
    run([python, "-m", "pip", "check"], "Checking dependency consistency")
    run([python, "-m", "acestep_modly.health", "--accelerator", lane["accelerator"]], "Checking real imports, kernels and WAV output before downloading weights")
    run([python, "-m", "acestep_modly.provision", "--models-root", models], "Provisioning/reusing the pinned model snapshot")
    state = {"extension_root": str(ROOT), "models_root": str(models), "upstream_revision": UPSTREAM_REVISION, "lane": lane}
    with tempfile.NamedTemporaryFile(mode="w", dir=STATE_FILE.parent, encoding="utf-8", delete=False) as file:
        json.dump(state, file, indent=2)
        temporary = Path(file.name)
    os.replace(temporary, STATE_FILE)
    log("Setup complete. Weights are outside the extension and survive updates. Run Text to Music in Workflows.")


def main():
    configure_stdio()
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        setup(parse_args(sys.argv))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        import traceback
        traceback.print_exc()
        print(f"[ACE-Step setup] ERROR: {type(exc).__name__}: {exc}. Setup did not complete. Check the preceding logs and run Repair; verified weights are retained.", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
