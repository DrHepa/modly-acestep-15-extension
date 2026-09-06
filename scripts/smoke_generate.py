"""Run a real 10-second generation through the Modly process protocol after setup."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--temp", required=True)
    parser.add_argument("--models-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--planner", action="store_true")
    args = parser.parse_args()
    request = {
        "nodeId": "text-to-music", "input": {"text": "Warm acoustic jazz, brushed drums, mellow piano, relaxed evening atmosphere"},
        "params": {"duration": 10, "seed": 42, "instrumental": "true", "device": args.device, "thinking": str(args.planner).lower()},
        "workspaceDir": args.workspace, "tempDir": args.temp,
    }
    if args.models_dir:
        request["models_dir"] = args.models_dir
    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "processor.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8", cwd=ROOT)
    child.stdin.write(json.dumps(request) + "\n")
    child.stdin.close()
    terminals = []
    for line in child.stdout:
        message = json.loads(line)
        print(line.rstrip(), flush=True)
        if message["type"] in ("done", "error"):
            terminals.append(message)
    status = child.wait()
    if status != 0 or len(terminals) != 1 or terminals[0]["type"] != "done":
        raise RuntimeError("Real generation failed; inspect the streamed logs")
    path = Path(terminals[0]["result"]["filePath"])
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError("Processor did not return an existing absolute output")
    print(f"Real smoke passed: {path}", flush=True)


if __name__ == "__main__":
    main()
