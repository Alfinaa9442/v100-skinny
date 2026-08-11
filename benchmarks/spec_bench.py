#!/usr/bin/env python3
"""Throughput benchmark for n-gram speculative decoding on the NVFP4 server.

Measures decode tokens/s on two workloads (natural generation vs
extraction-style, where prompt-lookup drafting has high acceptance) and
scrapes draft/acceptance counters from /metrics.
Baselines (no speculation): custom kernel 67.7 tok/s, Marlin 40.5 tok/s.
"""
import json
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.5-27b-nvfp4-skinny"


def chat(content, max_tokens, timeout=900):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_completion_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"]["content"], d["usage"],
            time.time() - t0)


def decode_rate(prompt, max_tokens):
    _, _, t1 = chat(prompt, 1)                      # prefill + 1 token
    out, usage, tfull = chat(prompt, max_tokens)
    gen = usage["completion_tokens"] - 1
    rate = gen / max(tfull - t1, 1e-6)
    return rate, usage["completion_tokens"], out


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for key in ("spec_decode_num_draft_tokens_total",
                "spec_decode_num_accepted_tokens_total",
                "spec_decode_num_drafts_total"):
        m = re.findall(rf"vllm:{key}\S*\s+([0-9.e+]+)", text)
        if m:
            out[key] = sum(float(v) for v in m)
    return out


# correctness with speculation on
for q, need in [("Count from 1 to 10, comma-separated.", "8, 9, 10"),
                ("List the first five prime numbers, comma-separated.", "11")]:
    reply, _, _ = chat(q, 48)
    ok = need in reply
    print(f"[{'PASS' if ok else 'FAIL'}] {q[:30]}  -> {reply.strip()[:60]!r}")

m0 = metrics()

passage = ("The lighthouse keeper checked the lamp at dusk. "
           "Salt wind pressed against the tower glass. "
           "Ships passed far out on the dark water. ") * 22

print("\nnatural generation (256 tok):")
rate, n, _ = decode_rate("Write a long story about a lighthouse keeper.", 256)
print(f"  {rate:.1f} tok/s decode ({n} tokens)")

print("extraction workload (repeat passage verbatim, ~512 tok):")
rate, n, out = decode_rate(
    "Repeat the following passage exactly, word for word:\n\n" + passage, 512)
print(f"  {rate:.1f} tok/s decode ({n} tokens)")
print(f"  output starts: {out.strip()[:70]!r}")

m1 = metrics()
drafts = m1.get("spec_decode_num_draft_tokens_total", 0) - \
    m0.get("spec_decode_num_draft_tokens_total", 0)
accepted = m1.get("spec_decode_num_accepted_tokens_total", 0) - \
    m0.get("spec_decode_num_accepted_tokens_total", 0)
if drafts:
    print(f"\ndraft tokens: {drafts:.0f}, accepted: {accepted:.0f} "
          f"({100*accepted/drafts:.1f}% acceptance)")
else:
    print("\nno spec-decode counters found in /metrics")
print("\nbaselines without speculation: custom 67.7 tok/s, marlin 40.5 tok/s")
