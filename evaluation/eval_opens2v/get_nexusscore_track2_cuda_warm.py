# Adapted from the benchmark working tree for portable evaluation.
import os
import sys
import runpy
from pathlib import Path

# Must happen before importing get_nexusscore_track2.py / mmengine / mmdet / mmyolo.
import torch


def main():
    print(f"[WRAPPER] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[WRAPPER] torch={torch.__version__}, torch_cuda={torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError("[WRAPPER] torch.cuda.is_available() is False")

    # With CUDA_VISIBLE_DEVICES=4, the visible device inside this process is cuda:0.
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    print(f"[WRAPPER] CUDA warmup OK: device={torch.cuda.get_device_name(0)}")

    target = Path(__file__).with_name("get_nexusscore_track2.py")

    # Preserve original CLI args, but execute the real script.
    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()