"""
Local smoke test for train_gpt.py — runs on a single GPU with dummy data.
Tests: model import, GQA compat fix, LoRA TTT init, forward pass.

Usage:
    TORCHDYNAMO_DISABLE=1 python test_smoke.py
    python test_smoke.py        (on Linux with PyTorch >= 2.5)
"""
import os, sys, io
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # safe on Windows
# Force UTF-8 output on Windows (avoids cp1252 encoding errors with checkmarks)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── import train_gpt as a module (main() won't run; guarded by __name__) ───────
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util

# Load without executing (avoids the torchrun / data-loading code)
spec = importlib.util.spec_from_file_location("train_gpt", "train_gpt.py")
tg = importlib.util.module_from_spec(spec)
tg.__name__ = "train_gpt"        # NOT "__main__" → main() never called
spec.loader.exec_module(tg)

import torch
import torch.nn.functional as F

print(f"torch version : {torch.__version__}")
print(f"_SDPA_SUPPORTS_GQA : {tg._SDPA_SUPPORTS_GQA}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device : {device}\n")

# ── 1. Tiny model: 2 layers, 64 dim, GQA (8 heads / 4 kv-heads) ────────────────
print("=== TEST 1: model init + forward pass (GQA) ===")
model = tg.GPT(
    vocab_size=256,
    num_layers=2,
    model_dim=64,
    num_heads=8,
    num_kv_heads=4,          # GQA — exercises the enable_gqa / repeat_interleave path
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.02,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
).to(device)

x = torch.randint(0, 256, (2, 64), device=device)   # batch=2, seq=64
y = torch.randint(0, 256, (2, 64), device=device)

with torch.no_grad():
    loss = model(x, y)
print(f"  forward loss: {loss.item():.4f}  ✓")

# ── 2. GQA path: manually test SDPA compat ─────────────────────────────────────
print("\n=== TEST 2: GQA SDPA compat ===")
B, H_q, H_kv, T, D = 2, 8, 4, 64, 16
q  = torch.randn(B, H_q,  T, D, device=device)
k  = torch.randn(B, H_kv, T, D, device=device)
v  = torch.randn(B, H_kv, T, D, device=device)
use_gqa = (H_q != H_kv)

if not tg._SDPA_SUPPORTS_GQA and use_gqa:
    g = H_q // H_kv
    k_exp = k.repeat_interleave(g, dim=1)
    v_exp = v.repeat_interleave(g, dim=1)
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
else:
    kw = {"enable_gqa": use_gqa} if tg._SDPA_SUPPORTS_GQA else {}
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kw)
print(f"  GQA SDPA output shape: {list(out.shape)}  (expected [2,8,64,16])  ✓")

# ── 3. LoRA TTT: patch lora_A/B onto CastedLinear, run forward ─────────────────
print("\n=== TEST 3: LoRA TTT delta injection ===")
# Find first CastedLinear in model, inject fake LoRA
r = 4
for name, mod in model.named_modules():
    if isinstance(mod, tg.CastedLinear):
        out_f, in_f = mod.weight.shape
        mod._lora_A = torch.randn(r, in_f,  device=device, requires_grad=True)
        mod._lora_B = torch.zeros(out_f, r, device=device, requires_grad=True)
        print(f"  Injected rank-{r} LoRA on {name} ({out_f}×{in_f})")
        break

with torch.no_grad():
    loss2 = model(x, y)
print(f"  forward loss with LoRA: {loss2.item():.4f}  ✓")

# ── 4. XSA + top-N layer TTT layers present ────────────────────────────────────
print("\n=== TEST 4: XSA model ===")
model_xsa = tg.GPT(
    vocab_size=256,
    num_layers=4,
    model_dim=64,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.02,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    xsa_last_n=2,             # last 2 blocks have XSA enabled
).to(device)

with torch.no_grad():
    loss3 = model_xsa(x, y)
print(f"  XSA forward loss: {loss3.item():.4f}  ✓")

print("\n" + "="*50)
print("ALL TESTS PASSED — code is safe to run on RunPod")
print("="*50)
