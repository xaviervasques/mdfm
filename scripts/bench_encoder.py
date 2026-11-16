"""T0.9: the dense encoder on the RTX 5080, measured before any model code depends on guesses.

Three measurements (docs/MFM_DESIGN_NOTE_ARCHITECTURE_2026-09-24.md, sections 4, 6, 7):
1. Model FLOP utilization (MFU) of a dense pre-norm transformer encoder, bf16, fused
   scaled-dot-product attention, on packed batches of 10 s windows whose lengths follow the
   corpus shape (mostly short single-sensor windows, some long multimodal ones). Two sizes:
   the design encoder (12 layers, width 768) and a pilot-size encoder (12 layers, width 512).
   MFU counts useful FLOPs only (6 x parameters x tokens plus attention over attended pairs)
   against two peaks: the nominal dense bf16 peak of the RTX 5080 (112.6 TFLOPS, vendor
   figure as quoted by the architecture study) and the matrix-product peak measured on this
   GPU by scripts/check_env.py.
2. Attention backends under a boolean typed and block-diagonal mask: math, memory-efficient,
   flash (expected to refuse an arbitrary mask), cuDNN, and FlexAttention with a block mask
   (needs Triton, often absent on Windows). Forward plus backward time per call.
3. The one-way rule as an autograd test: with the typed mask, the gradient of every body
   output with respect to every nuisance and neural input token must be exactly zero, in a
   packed batch and through every layer; and a control without the mask must be non-zero.

Writes reports/bench/bench_encoder.json and bench_encoder.md. Needs the mdfm environment:
<conda>/envs/mdfm/python.exe scripts/bench_encoder.py
"""
import json
import math
import os
import sys
import time

# torch._dynamo, imported by the optimizers, derives its cache folder from getpass.getuser(),
# which needs the Unix module pwd when no user variable is set (the case in the session's
# Git Bash); an explicit cache folder in the gitignored cache/ avoids it.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "torchinductor"))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(here, "reports", "bench")
os.makedirs(out_dir, exist_ok=True)
assert torch.cuda.is_available(), "needs the GPU"
dev = torch.device("cuda")
torch.manual_seed(0)
NOMINAL_PEAK = 112.6e12
env_json = os.path.join(here, "reports", "env", "check_env.json")
MEASURED_PEAK = json.load(open(env_json, encoding="utf-8")).get("bf16_matmul_tflops", 0) * 1e12 if os.path.exists(env_json) else 0

BODY, NUIS, NEURAL = 0, 1, 2


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, layers, d, h):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        for blk in self.blocks:
            x = blk(x, mask)
        return self.norm(x)


def window_lengths(total, gen):
    """Window lengths in tokens: 80 percent single-sensor windows (40 to 120 tokens), 15
    percent mid-size (200 to 600), 5 percent full multimodal (1,000 to 1,500)."""
    lens = []
    while sum(lens) < total:
        u = torch.rand(1, generator=gen).item()
        if u < 0.80:
            n = int(torch.randint(40, 121, (1,), generator=gen))
        elif u < 0.95:
            n = int(torch.randint(200, 601, (1,), generator=gen))
        else:
            n = int(torch.randint(1000, 1501, (1,), generator=gen))
        if sum(lens) + n > total:
            break
        lens.append(n)
    return lens


def packed_mask(lens, total, types=None):
    """Block-diagonal boolean mask (True = may attend) for windows packed in one row of
    length total; padding tokens attend only to themselves. With types, body queries may not
    attend to nuisance or neural keys."""
    seg = torch.full((total,), -1, dtype=torch.long)
    pos = 0
    for i, n in enumerate(lens):
        seg[pos:pos + n] = i
        pos += n
    m = seg[:, None] == seg[None, :]
    m &= seg[:, None] >= 0
    m |= torch.eye(total, dtype=torch.bool)
    if types is not None:
        body_q = types[:, None] == BODY
        other_k = types[None, :] != BODY
        m &= ~(body_q & other_k)
        m |= torch.eye(total, dtype=torch.bool)
    return m, seg


def useful_flops(n_params, lens, layers, d):
    tokens = sum(lens)
    attn_pairs = sum(n * n for n in lens)
    # training = 3 x forward; forward = 2 x params x tokens + 4 x d x attended pairs per layer
    return 3 * (2 * n_params * tokens + 4 * d * attn_pairs * layers), tokens


def bench_mfu(layers, d, h, row=4096, rows=4, steps=12):
    model = Encoder(layers, d, h).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(1)
    batches = []
    for _ in range(steps):
        masks, all_lens = [], []
        for _ in range(rows):
            lens = window_lengths(row, gen)
            m, _ = packed_mask(lens, row)
            masks.append(m)
            all_lens += lens
        batches.append((torch.stack(masks)[:, None].to(dev), all_lens))
    x = torch.randn(rows, row, d, device=dev)
    for i, (mask, _) in enumerate(batches[:3]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, mask).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    flops, toks = 0, 0
    t = time.perf_counter()
    for mask, lens in batches[3:]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, mask).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        f, n = useful_flops(n_params, lens, layers, d)
        flops += f
        toks += n
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    rate = flops / dt
    res = {"layers": layers, "width": d, "heads": h, "params": n_params, "rows": rows, "row_tokens": row,
           "useful_tokens_per_s": round(toks / dt), "useful_tflops": round(rate / 1e12, 1),
           "mfu_nominal_peak": round(rate / NOMINAL_PEAK, 3),
           "mfu_measured_matmul_peak": round(rate / MEASURED_PEAK, 3) if MEASURED_PEAK else None,
           "padding_fraction": round(1 - toks / (rows * row * (steps - 3)), 3),
           "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del model, opt
    torch.cuda.empty_cache()
    return res


def bench_mfu_bucketed(layers, d, h, tokens_per_step=4096, steps=24):
    """Length bucketing instead of long packed rows: each step draws one bucket with the corpus
    shape (80 percent short windows in rows of 128 tokens, 15 percent mid in rows of 640, 5
    percent long in rows of 1,536), packs windows of that bucket only, and keeps the
    block-diagonal mask per row. Attention then spends little on masked pairs."""
    model = Encoder(layers, d, h).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(4)
    buckets = [(0.80, 128, (40, 120)), (0.15, 640, (200, 600)), (0.05, 1536, (1000, 1500))]
    plan = []
    for _ in range(steps):
        u = torch.rand(1, generator=gen).item()
        acc = 0
        for p_b, row, (lo, hi) in buckets:
            acc += p_b
            if u < acc:
                break
        rows = max(1, tokens_per_step // row)
        masks, lens_all = [], []
        for _ in range(rows):
            lens, used = [], 0
            while True:
                n = int(torch.randint(lo, hi + 1, (1,), generator=gen))
                if used + n > row:
                    break
                lens.append(n)
                used += n
            m, _ = packed_mask(lens, row)
            masks.append(m)
            lens_all += lens
        plan.append((row, rows, torch.stack(masks)[:, None].to(dev), lens_all))
    xs = {}
    for row, rows, _, _ in plan:
        xs.setdefault((row, rows), torch.randn(rows, row, d, device=dev))
    for row, rows, mask, _ in plan[:3]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(xs[(row, rows)], mask).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    flops, toks, slots = 0, 0, 0
    t = time.perf_counter()
    for row, rows, mask, lens in plan[3:]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(xs[(row, rows)], mask).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        f, n = useful_flops(n_params, lens, layers, d)
        flops += f
        toks += n
        slots += row * rows
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    rate = flops / dt
    res = {"layers": layers, "width": d, "params": n_params, "mode": "bucketed",
           "useful_tokens_per_s": round(toks / dt), "useful_tflops": round(rate / 1e12, 1),
           "mfu_nominal_peak": round(rate / NOMINAL_PEAK, 3),
           "mfu_measured_matmul_peak": round(rate / MEASURED_PEAK, 3) if MEASURED_PEAK else None,
           "padding_fraction": round(1 - toks / slots, 3)}
    del model, opt
    torch.cuda.empty_cache()
    return res


def bench_backends(d=768, h=12, row=4096, reps=10):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    gen = torch.Generator().manual_seed(2)
    lens = window_lengths(row, gen)
    types = torch.randint(0, 3, (row,), generator=gen)
    m, seg = packed_mask(lens, row, types)
    mask = m[None, None].to(dev)
    q = torch.randn(4, h, row, d // h, device=dev, dtype=torch.bfloat16, requires_grad=True)
    out = {}
    ref = None
    for name, be in (("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
        try:
            with sdpa_kernel([be]):
                y = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
                y.float().sum().backward()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(reps):
                    y = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
                    y.float().sum().backward()
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) / reps * 1000
            if ref is None:
                ref = y.detach().float()
            err = (y.detach().float() - ref).abs().max().item()
            out[name] = {"ok": True, "fwd_bwd_ms": round(ms, 2), "max_abs_diff_vs_math": round(err, 4)}
        except Exception as e:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        seg_d, types_d = seg.to(dev), types.to(dev)

        def mod(b, hh, qi, ki):
            same = (seg_d[qi] == seg_d[ki]) & (seg_d[qi] >= 0)
            allowed = same & ~((types_d[qi] == BODY) & (types_d[ki] != BODY))
            return allowed | (qi == ki)

        bm = create_block_mask(mod, None, None, row, row, device="cuda")
        fa = torch.compile(flex_attention)
        y = fa(q, q, q, block_mask=bm)
        y.float().sum().backward()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            y = fa(q, q, q, block_mask=bm)
            y.float().sum().backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / reps * 1000
        err = (y.detach().float() - ref).abs().max().item() if ref is not None else None
        out["flex"] = {"ok": True, "fwd_bwd_ms": round(ms, 2), "max_abs_diff_vs_math": round(err, 4) if err is not None else None}
    except Exception as e:  # noqa: BLE001
        out["flex"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    out["attended_fraction"] = round(m.float().mean().item(), 4)
    return out


def one_way_test(layers=4, d=256, h=4, row=1024):
    model = Encoder(layers, d, h).to(dev).float()
    gen = torch.Generator().manual_seed(3)
    lens = window_lengths(row, gen)
    types = torch.randint(0, 3, (row,), generator=gen)
    x = torch.randn(1, row, d, device=dev, requires_grad=True)
    res = {}
    for label, typed in (("typed_mask", True), ("control_no_type_rule", False)):
        m, seg = packed_mask(lens, row, types if typed else None)
        x.grad = None
        y = model(x, m[None, None].to(dev))
        body = (types == BODY) & (seg >= 0)
        y[0, body.to(dev)].sum().backward()
        g = x.grad[0].abs().sum(-1).cpu()
        other = (types != BODY) & (seg >= 0)
        res[label] = {"max_grad_on_nuisance_and_neural_inputs": float(g[other].max()),
                      "exactly_zero": bool((g[other] == 0).all()), "body_tokens": int(body.sum()),
                      "other_tokens": int(other.sum())}
    res["pass"] = res["typed_mask"]["exactly_zero"] and not res["control_no_type_rule"]["exactly_zero"]
    return res


report = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
          "nominal_bf16_peak_tflops": NOMINAL_PEAK / 1e12, "measured_matmul_peak_tflops": MEASURED_PEAK / 1e12}
# Order matters: a failed FlexAttention import (torch.compile needs the Unix module pwd on
# this Windows build) leaves torch._dynamo half-imported, which breaks later optimizer
# construction; so the utilization runs come first and the backend test, with Flex, last.
report["mfu"] = []
for layers, d, h in ((12, 512, 8), (12, 768, 12)):
    print(f"MFU {layers}x{d} ...", flush=True)
    r = bench_mfu(layers, d, h)
    report["mfu"].append(r)
    print(json.dumps(r, indent=1), flush=True)
    r = bench_mfu_bucketed(layers, d, h)
    report["mfu"].append(r)
    print(json.dumps(r, indent=1), flush=True)
print("one-way test ...", flush=True)
report["one_way"] = one_way_test()
print(json.dumps(report["one_way"], indent=1), flush=True)
print("attention backends ...", flush=True)
report["backends"] = bench_backends()
print(json.dumps(report["backends"], indent=1), flush=True)

json.dump(report, open(os.path.join(out_dir, "bench_encoder.json"), "w", encoding="utf-8", newline="\n"), indent=1)
md = ["# Dense encoder on the RTX 5080 (T0.9)", "",
      f"Measured by scripts/bench_encoder.py with torch {report['torch']} on {report['device']}.", "",
      "## One-way rule, autograd test", "",
      f"Typed mask: largest gradient of body outputs on nuisance and neural inputs = "
      f"{report['one_way']['typed_mask']['max_grad_on_nuisance_and_neural_inputs']} "
      f"(exactly zero: {report['one_way']['typed_mask']['exactly_zero']}); control without the type rule: "
      f"{report['one_way']['control_no_type_rule']['max_grad_on_nuisance_and_neural_inputs']:.3g}. "
      f"Pass: {report['one_way']['pass']}.", "",
      "## Attention backends, packed windows with the typed mask (4 x 12 heads x 4,096 tokens)", "",
      "| backend | runs | forward and backward, ms | max difference to math |", "|---|---|---|---|"]
for k, v in report["backends"].items():
    if isinstance(v, dict):
        md.append(f"| {k} | {'yes' if v['ok'] else 'no: ' + v['error']} | {v.get('fwd_bwd_ms', '')} | {v.get('max_abs_diff_vs_math', '')} |")
md += ["", f"Attended fraction of the mask: {report['backends']['attended_fraction']}.", "",
       "## Model FLOP utilization, packed training steps", "",
       "| encoder | batching | parameters | useful tokens/s | useful TFLOPS | MFU on nominal peak | MFU on measured matmul peak | padding | peak memory GB |",
       "|---|---|---|---|---|---|---|---|---|"]
for r in report["mfu"]:
    md.append(f"| {r['layers']} x {r['width']} | {r.get('mode', 'rows of 4,096')} | {r['params']:,} | {r['useful_tokens_per_s']:,} | "
              f"{r['useful_tflops']} | {r['mfu_nominal_peak']} | {r['mfu_measured_matmul_peak']} | {r['padding_fraction']} | "
              f"{r.get('peak_memory_gb', '')} |")
text = "\n".join(md) + "\n"
assert chr(0x2014) not in text
open(os.path.join(out_dir, "bench_encoder.md"), "w", encoding="utf-8", newline="\n").write(text)
print(text)
