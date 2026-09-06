"""Read official wheel indices for all supported native lanes; no wheels downloaded."""
import concurrent.futures
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acestep_modly.platforms import select_lane, torch_requirements


def fetch_index(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        page = response.read().decode("utf-8")
    return url, [urllib.parse.unquote(u).split("/")[-1].split("#")[0] for u in re.findall(r'href="([^"]+)"', page)]


def main():
    checks, indices = [], set()
    for system, arch in (("win32", "x64"), ("linux", "x64"), ("linux", "arm64")):
        for version in ((3, 11), (3, 12)):
            for accelerator in ("cpu", "cuda"):
                lane = select_lane({"accelerator": accelerator}, system=system, machine=arch, version=version, driver_cuda=130)
                for requirement in torch_requirements(lane):
                    name, package_version = requirement.split("==")
                    url = lane["index"] + "/" + name + "/"
                    indices.add(url)
                    checks.append((lane, name, package_version, url))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pages = dict(pool.map(fetch_index, sorted(indices)))
    results = []
    for lane, name, version, url in checks:
        python_tag = "cp" + lane["python"].replace(".", "")
        platform_tag = "win_amd64" if lane["system"] == "win32" else ("aarch64" if lane["arch"] == "arm64" else "x86_64")
        prefix = f"{name}-{version}-{python_tag}-{python_tag}-"
        files = [f for f in pages[url] if f.startswith(prefix) and f.endswith(platform_tag + ".whl")]
        results.append({"target": f"{lane['system']}/{lane['arch']}/{python_tag}/{lane['flavor']}", "package": name, "available": bool(files), "files": files})
    print(json.dumps(results, indent=2))
    return int(not all(r["available"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
