"""Print + save environment info (python, torch, CUDA, GPU, VRAM).

Run:  python -m metalai.common.env_check
Writes reports/env.json
"""
from __future__ import annotations

import platform
import sys

from metalai.common.utils import REPORTS, save_json


def main() -> None:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["gpu_vram_gb"] = round(p.total_memory / 1024**3, 2)
            x = torch.randn(1024, 1024, device="cuda")
            info["gpu_matmul_ok"] = bool((x @ x).isfinite().all().item())
    except ImportError:
        info["torch"] = "NOT INSTALLED"
    for mod in ["torchvision", "ultralytics", "sklearn", "skimage", "cv2", "pytorch_grad_cam", "gradio"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"MISSING ({type(e).__name__})"
    for k, v in info.items():
        print(f"{k:>18}: {v}")
    save_json(info, REPORTS / "env.json")
    print(f"\nSaved {REPORTS / 'env.json'}")


if __name__ == "__main__":
    main()
