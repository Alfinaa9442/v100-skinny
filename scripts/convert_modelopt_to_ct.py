#!/usr/bin/env python3
"""Convert nvidia/Qwen3.6-27B-NVFP4 (ModelOpt mixed FP8+NVFP4) to a
compressed-tensors all-NVFP4 checkpoint servable by our SM70 route.

- W4A16_NVFP4 layers (MLPs): tensor rename + global-scale inversion
  (CT stores divisors); packed nibbles kept as exported.
- FP8 layers (attn/GDN projections): dequantize (exact) then requantize
  to NVFP4 group-16 with our packer. Provenance note: these layers
  deviate from the official FP8 quantization by construction.
- Everything else (norms, embeddings, lm_head, MTP head, conv/gates):
  copied unchanged.
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NV = os.path.expanduser("~/models/Qwen3.6-27B-NVFP4")
CT_REF = os.path.expanduser("~/models/Qwen3.6-27B-NVFP4-CT")  # config donor
OUT = os.path.expanduser(os.environ.get("CONV_OUT", "~/models/Qwen3.6-27B-NVFP4-CTfull"))
os.makedirs(OUT, exist_ok=True)
dev = torch.device(os.environ.get("CONV_DEV", "cuda:0"))

MAGS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
MIDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)


def pack_nvfp4(w):
    n, k = w.shape
    wf = w.float().view(n, k // 16, 16)
    scale = wf.abs().amax(-1) / 6.0
    g = float(scale.max().item() / 448.0) or 1e-12
    q8 = (scale / g).to(torch.float8_e4m3fn)
    eff = (q8.float() * g).clamp(min=1e-12).unsqueeze(-1)
    idx = torch.bucketize((wf / eff).abs(), MIDS)
    sgn = (wf / eff) < 0
    codes = (idx | (sgn.long() << 3)).view(n, k).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed.cpu(), q8.view(torch.uint8).view(n, k // 16).cpu(), g


qcfg = json.load(open(os.path.join(NV, "config.json")))["quantization_config"]
algo = {m: v.get("quant_algo") for m, v in qcfg["quantized_layers"].items()}
fp8_mods = sorted(m for m, a in algo.items() if a == "FP8")
nv4_mods = sorted(m for m, a in algo.items() if a == "W4A16_NVFP4")
print(f"{len(fp8_mods)} FP8 modules to requantize, "
      f"{len(nv4_mods)} NVFP4 modules to rename")

idx_map = json.load(open(os.path.join(NV,
                    "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(idx_map.values()))
raw = {}
for shard in shards:
    with safe_open(os.path.join(NV, shard), framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
print(f"loaded {len(raw)} tensors")

out = {}
requant_stats = []

# lm_head: NVIDIA ships it NVFP4-quantized, but vLLM's ParallelLMHead
# loads only a plain weight. Dequantize to bf16 here; the runtime 4-bit
# lm_head wrapper (session default) re-packs it at load.
if "lm_head" in nv4_mods:
    nv4_mods.remove("lm_head")
    MAGS_CPU = MAGS.cpu()
    q = raw.pop("lm_head.weight")
    sc = raw.pop("lm_head.weight_scale")
    g2 = raw.pop("lm_head.weight_scale_2").float()
    raw.pop("lm_head.input_scale", None)
    rows = []
    step = 16384
    for i in range(0, q.shape[0], step):
        qd = q[i:i + step].to(dev)
        lo = (qd & 0x0F).long()
        hi = (qd >> 4).long()
        idx4 = torch.stack([lo, hi], -1).view(qd.shape[0], qd.shape[1] * 2)
        mag = MAGS[idx4 & 7]
        sgn = torch.where((idx4 & 8) > 0, -1.0, 1.0)
        eff = (sc[i:i + step].to(dev).float() * g2.item()
               ).repeat_interleave(16, dim=1)
        rows.append((mag * sgn * eff).to(torch.bfloat16).cpu())
        del qd, lo, hi, idx4, mag, sgn, eff
        torch.cuda.empty_cache()
    out["lm_head.weight"] = torch.cat(rows).contiguous()
    del q, sc, rows
    print("lm_head dequantized to bf16 for loader compatibility")

for mod in nv4_mods:
    out[mod + ".weight_packed"] = raw.pop(mod + ".weight")
    out[mod + ".weight_scale"] = raw.pop(mod + ".weight_scale")
    ws2 = raw.pop(mod + ".weight_scale_2").float()
    out[mod + ".weight_global_scale"] = (1.0 / ws2)
    in_s = raw.pop(mod + ".input_scale").float()
    out[mod + ".input_global_scale"] = (1.0 / in_s)

FP8_MODE = os.environ.get("CONV_FP8_MODE", "requant")
for mod in fp8_mods:
    w8 = raw.pop(mod + ".weight")
    ws = raw.pop(mod + ".weight_scale").float()
    in_s = raw.pop(mod + ".input_scale").float()
    if FP8_MODE == "fp16":
        # lossless: fp8 e4m3 -> fp16 exact, times the per-tensor scale
        out[mod + ".weight"] = (w8.float() * ws.item()).half()
    else:
        w = (w8.to(dev).float() * ws.item()).half()
        packed, scales, g = pack_nvfp4(w)
        out[mod + ".weight_packed"] = packed
        out[mod + ".weight_scale"] = scales.view(torch.float8_e4m3fn)
        out[mod + ".weight_global_scale"] = torch.tensor(1.0 / g)
        out[mod + ".input_global_scale"] = (1.0 / in_s)
        del w
        torch.cuda.empty_cache()
    requant_stats.append(mod)
    del w8
print(f"processed {len(requant_stats)} FP8 modules (mode={FP8_MODE})")

out.update(raw)  # untouched tensors (bf16 heads, norms, mtp, etc.)
print(f"writing {len(out)} tensors")
save_file(out, os.path.join(OUT, "model.safetensors"),
          metadata={"format": "pt"})

# Config: CT-format config (same arch, booted successfully) with the
# quantization ignore list rebuilt from what we actually quantized.
# Source preference: a local donor CT checkpoint if present, else the
# in-repo template (scripts/ct_config_template.json) — so the only
# external artifact needed is the NVIDIA source checkpoint.
_donor_cfg = os.path.join(CT_REF, "config.json")
_template_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "ct_config_template.json")
cfg_path = _donor_cfg if os.path.exists(_donor_cfg) else _template_cfg
print(f"config base: {cfg_path}")
cfg = json.load(open(cfg_path))
donor_q = cfg["quantization_config"]
quantized = set(nv4_mods) | set(fp8_mods)
lin_suffixes = (".weight_packed",)
all_linear_prefixes = {k[:-len(".weight_packed")] for k in out
                      if k.endswith(".weight_packed")}
assert all_linear_prefixes == quantized
donor_q["ignore"] = sorted(qcfg.get("ignore", []))
cfg["quantization_config"] = donor_q
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
print("ignore list entries:", len(donor_q["ignore"]))

for fn in ("generation_config.json", "tokenizer.json",
           "tokenizer_config.json", "chat_template.jinja"):
    for src_dir in (CT_REF, NV):
        p = os.path.join(src_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, OUT)
            break
size = os.path.getsize(os.path.join(OUT, "model.safetensors")) / 2**30
print(f"DONE: {OUT} ({size:.1f} GB)")
