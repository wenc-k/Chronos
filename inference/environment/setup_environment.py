#!/usr/bin/env python3
"""Fetch pinned official sources and install into the active backend environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def command(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def checkout(profile, root):
    revision = profile["revision"]
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=root.name + "-clone-", dir=root.parent))
        try:
            command(["git", "clone", "--no-checkout", profile["repository"], staging])
            command(["git", "-C", staging, "checkout", "--detach", revision])
            staging.rename(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if git_output(root, "rev-parse", "HEAD") != revision:
        raise SystemExit(f"Expected revision {revision} at {root}. Select a clean target directory.")
    if git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"Tracked upstream files have local changes at {root}. Select a clean target directory.")
    return root


def main():
    profiles = json.loads((HERE / "sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=profiles)
    parser.add_argument("--repo-root", type=Path, help="Destination for the selected official checkout.")
    parser.add_argument("--fetch-only", action="store_true", help="Fetch sources without changing Python packages.")
    args = parser.parse_args()
    profile = profiles[args.backend]
    root = (args.repo_root or HERE.parent / ".deps" / profile["directory"]).resolve()
    if not args.fetch_only:
        actual = f"{sys.version_info.major}.{sys.version_info.minor}"
        if actual != profile["python"]:
            parser.error(f"This profile requires Python {profile['python']}, found {actual}.")
        if sys.prefix == sys.base_prefix and not os.environ.get("CONDA_PREFIX"):
            parser.error("Activate a dedicated Conda environment or venv first.")
    checkout(profile, root)
    if args.fetch_only:
        print(root)
        return
    constraints = HERE / f"requirements-{args.backend}.txt"
    pip = [sys.executable, "-m", "pip"]
    command(pip + ["install", "--index-url", profile["torch_index"],
                   *(name + "==" + profile[name] for name in ("torch", "torchvision", "torchaudio"))])
    command(pip + ["install", "setuptools", "wheel", "packaging", "ninja"])
    if args.backend == "wan22":
        # Install non-build dependencies before the FlashAttention build imports torch.
        official = (root / "requirements.txt").read_text().splitlines()
        ordinary = [line for line in official if not line.strip().startswith("flash_attn")]
        with tempfile.TemporaryDirectory(prefix="wan22-requirements-") as scratch:
            requirements = Path(scratch) / "requirements.txt"
            requirements.write_text("\n".join(ordinary) + "\n")
            command(pip + ["install", "-c", constraints, "-r", requirements, "-r", constraints])
        command(pip + ["install", "--no-build-isolation", "-c", constraints,
                       "flash-attn==" + profile["flash_attn"]])
    else:
        command(pip + ["install", "-c", constraints, "-r", root / "requirements.txt", "-r", constraints])
    command(pip + ["check"])
    print("Prepared", args.backend, "at", root)
    print("Run check_environment.py in an allocated GPU session before generation.")


if __name__ == "__main__":
    main()
