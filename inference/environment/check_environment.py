#!/usr/bin/env python3
"""Inspect the active environment; optional import and small CUDA-kernel checks."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    profiles = json.loads((HERE / "sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=profiles)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--gpu-check", action="store_true",
                        help="Run a small kernel in an allocated GPU session.")
    args = parser.parse_args()
    profile = profiles[args.backend]
    failures = []
    expected_python = profile["python"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        failures.append(f"Python {actual_python}; expected {expected_python}")
    for line in (HERE / f"requirements-{args.backend}.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        package, expected = line.split("==", 1)
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+", 1)[0] != expected:
            failures.append(f"{package} {actual}; expected {expected}")
    root = (args.repo_root or HERE.parent / ".deps" / profile["directory"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != profile["revision"]:
        failures.append(f"Source revision {revision}; expected {profile['revision']}")
    if args.imports or args.gpu_check:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        def offline(event, arguments):
            if event in ("socket.connect", "socket.getaddrinfo"):
                raise RuntimeError("Environment checks run offline.")
        sys.addaudithook(offline)
        sys.path.insert(0, str(root))
        import torch
        expected_cuda = profile["torch_index"].rsplit("cu", 1)[-1]
        expected_cuda = expected_cuda[:2] + "." + expected_cuda[2:]
        if torch.version.cuda != expected_cuda:
            failures.append(f"PyTorch CUDA {torch.version.cuda}; expected {expected_cuda}")
        if args.backend == "wan22":
            import flash_attn
            from diffusers.loaders import PeftAdapterMixin
            from peft import LoraConfig, set_peft_model_state_dict
            from transformers import Cache, DynamicCache, EncoderDecoderCache
            from wan import WanI2V
            if metadata.version("flash-attn") != profile["flash_attn"]:
                failures.append("FlashAttention version differs from sources.json")
            print("PASS Wan imports")
        else:
            # Import declarations without starting the web application or a model.
            import folder_paths
            import comfy.cli_args
            print("PASS ComfyUI base imports")
        if args.gpu_check:
            if not torch.cuda.is_available():
                raise SystemExit("CUDA is unavailable in this process.")
            if args.backend == "wan22":
                from flash_attn import flash_attn_func
                q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16)
                result = flash_attn_func(q, q, q)
            else:
                q = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
                result = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            if not torch.isfinite(result).all().item():
                raise SystemExit("CUDA-kernel check produced non-finite values.")
            print("PASS CUDA attention kernel")
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS configured versions and upstream revision")


if __name__ == "__main__":
    main()
