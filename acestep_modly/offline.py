"""Contain caches and disable network before importing upstream ML libraries."""
import os
import socket
import sys

from .constants import VENDOR


def configure(checkpoints, temporary):
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "HF_HOME": str(temporary / "hf"), "HF_HUB_CACHE": str(temporary / "hf" / "hub"),
        "HF_MODULES_CACHE": str(temporary / "hf" / "modules"),
        "TORCH_HOME": str(temporary / "torch"), "TOKENIZERS_PARALLELISM": "false",
        "ACESTEP_CHECKPOINTS_DIR": str(checkpoints), "ACESTEP_PROJECT_ROOT": str(temporary),
        "ACESTEP_DISABLE_TQDM": "1", "ACESTEP_VAE_CHECKPOINT": "official",
    })
    sys.path.insert(0, str(VENDOR))


def deny_network():
    """Defense in depth against any upstream automatic download fallback."""
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled during ACE-Step generation; use setup/Repair for weights")
    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
