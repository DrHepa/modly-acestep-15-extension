"""Install the real CPU lane for CI, deliberately never provision model weights."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acestep_modly.platforms import select_lane, torch_requirements
from acestep_modly.vendor import ensure_source


def main():
    lane = select_lane({"accelerator": "cpu"})
    ensure_source(print)
    subprocess.run([sys.executable, "-m", "pip", "install", *torch_requirements(lane), "--index-url", lane["index"]], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
