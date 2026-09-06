"""Immutable integration identity and audited upstream revisions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTENSION_ID = "modly-acestep-15-extension"
NODE_ID = "text-to-music"
UPSTREAM_REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
MODLY_REVISION = "1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65"
DIT_MODEL = "acestep-v15-turbo"
LM_MODEL = "acestep-5Hz-lm-1.7B"
VENDOR = ROOT / "vendor" / "ace-step"
STATE_FILE = ROOT / "venv" / "modly-acestep.json"
