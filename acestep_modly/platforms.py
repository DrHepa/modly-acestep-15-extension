"""Explicit upstream PyTorch lanes, selected from real host and driver evidence."""
import platform
import re
import subprocess
import sys


def normalize_arch(value):
    value = str(value).lower()
    return {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(value, value)


def probe_cuda_version():
    """Read driver-supported CUDA; current Modly caps its JSON value at 12.8."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    return int(match[1]) * 10 + int(match[2]) if match else 0


def select_lane(context, *, system=None, machine=None, version=None, driver_cuda=0):
    """Return a dependency lane; never mistake ARM64 for x86_64."""
    system = system or sys.platform
    machine = normalize_arch(machine or platform.machine())
    version = version or sys.version_info[:2]
    if tuple(version) not in ((3, 11), (3, 12)):
        raise ValueError("ACE-Step requires CPython 3.11 or 3.12")
    if (system, machine) not in (("win32", "x64"), ("linux", "x64"), ("linux", "arm64")):
        raise ValueError("Supported targets: Windows x64, Linux x64 and Linux ARM64")
    if context.get("platform", system) != system or normalize_arch(context.get("arch", machine)) != machine:
        raise ValueError("Setup platform/arch disagrees with the running Python interpreter")
    gpu_sm = int(context.get("gpu_sm", 0))
    accelerator = context.get("accelerator", "cuda" if gpu_sm > 0 else "cpu")
    if accelerator not in ("cpu", "cuda"):
        raise ValueError("This release supports CUDA and CPU, not ROCm/MPS/XPU")
    cuda = driver_cuda or int(context.get("cuda_version", 0))
    torch_version, vision_version = ("2.7.1", "0.22.1") if system == "win32" else ("2.10.0", "0.25.0")
    flavor = "cpu"
    if accelerator == "cuda":
        flavor, minimum = ("cu130", 130) if machine == "arm64" else ("cu128", 128)
        if cuda and cuda < minimum:
            raise ValueError(f"This {system}/{machine} lane requires a driver supporting CUDA {minimum // 10}.{minimum % 10}; detected {cuda}. Update the driver or explicitly select CPU.")
    return {
        "system": system, "arch": machine, "python": ".".join(map(str, version)),
        "accelerator": accelerator, "flavor": flavor,
        "torch": torch_version, "torchvision": vision_version,
        "index": f"https://download.pytorch.org/whl/{flavor}",
    }


def torch_requirements(lane):
    """Keep all three native PyTorch wheels on the same ABI/runtime lane."""
    suffix = "+" + lane["flavor"]
    return [f"torch=={lane['torch']}{suffix}", f"torchaudio=={lane['torch']}{suffix}", f"torchvision=={lane['torchvision']}{suffix}"]
