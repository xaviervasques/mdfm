"""Check the mdfm environment: versions, CUDA, the RTX 5080, bf16 matmul and masked attention.

Prints the interpreter, package versions, whether torch sees the GPU with compute
capability 12.0, then times a bf16 matrix product and scaled-dot-product attention with a
boolean mask (the typed-mask case of the design) on the GPU. Writes reports/env/check_env.json.
It is a smoke test, not the utilization benchmark of plan task T0.9.

Run: conda activate mdfm, then python scripts/check_env.py
"""
import importlib
import json
import os
import platform
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform()}
for name in ("torch", "numpy", "scipy", "pandas", "h5py", "sklearn", "statsmodels", "mne", "mne_bids",
             "pyedflib", "pyarrow", "yaml", "einops", "matplotlib"):
    try:
        out[name] = getattr(importlib.import_module(name), "__version__", "present")
    except Exception as e:  # noqa: BLE001
        out[name] = f"missing ({type(e).__name__})"

try:
    import torch
    import torch.nn.functional as F
    out["cuda_available"] = torch.cuda.is_available()
    out["torch_cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        out["device"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        out["arch_list"] = torch.cuda.get_arch_list()
        a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t = time.perf_counter()
        n = 20
        for _ in range(n):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / n
        out["bf16_matmul_tflops"] = round(2 * 4096 ** 3 / dt / 1e12, 1)
        # masked attention: batch 8, 12 heads, 1,500 tokens, head dim 64, boolean mask
        q = torch.randn(8, 12, 1500, 64, device=dev, dtype=torch.bfloat16)
        mask = torch.rand(1500, 1500, device=dev) > 0.1
        mask |= torch.eye(1500, device=dev, dtype=torch.bool)
        for _ in range(3):
            F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
        torch.cuda.synchronize()
        out["sdpa_masked_ms"] = round((time.perf_counter() - t) / 10 * 1000, 2)
        x = torch.randn(64, 768, device=dev, requires_grad=True)
        y = torch.nn.Linear(768, 768).to(dev)(x).sum()
        y.backward()
        out["autograd_ok"] = x.grad is not None
except Exception as e:  # noqa: BLE001
    out["torch_error"] = repr(e)

os.makedirs(os.path.join(here, "reports", "env"), exist_ok=True)
json.dump(out, open(os.path.join(here, "reports", "env", "check_env.json"), "w", encoding="utf-8", newline="\n"),
          indent=1)
print(json.dumps(out, indent=1))
ok = out.get("cuda_available") and out.get("capability") == [12, 0] and "torch_error" not in out
print("ENV OK" if ok else "ENV NOT READY")
sys.exit(0 if ok else 1)
