#!/usr/bin/env python3
"""Synthetic end-to-end pipeline test for the skinny-kernel NVFP4 server.

Exercises the full serving stack through the OpenAI API:
  1. health + model listing
  2. greedy correctness probes (known answers)
  3. short-prompt prefill (M<=64 -> skinny wmma) vs long-prompt prefill
     (M>64 -> marlin) route switching
  4. greedy determinism (two identical requests must match exactly)
  5. concurrency at max_num_seqs=2
  6. decode TPOT probe (two generation lengths isolate decode from prefill)

Waits for /health before starting. Exit 0 iff every check passes.
"""
import concurrent.futures as cf
import json
import os
import time
import urllib.request

BASE = os.environ.get("E2E_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("E2E_MODEL", "qwen3.5-27b-nvfp4-skinny")
# STRUCTURAL=1: skip semantic answer checks (tiny truncated models emit
# gibberish by design); still verifies routes/determinism/concurrency.
STRUCTURAL = os.environ.get("E2E_STRUCTURAL", "0") == "1"
results = []


def api(path, payload=None, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(content, max_tokens=48, timeout=600):
    t0 = time.time()
    d = api("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout)
    dt = time.time() - t0
    return d["choices"][0]["message"]["content"], d["usage"], dt


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# 0. wait for health
for _ in range(240):
    try:
        urllib.request.urlopen(BASE + "/health", timeout=2)
        break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit("server never became healthy")
print("server healthy\n")

# 1. model listing
models = [m["id"] for m in api("/v1/models")["data"]]
check("model listed", MODEL in models, str(models))

# 2. correctness probes (greedy, verifiable) — skipped in structural mode
if not STRUCTURAL:
    reply, usage, _ = chat("Answer with just the number: 2+2=?")
    check("arithmetic", "4" in reply, repr(reply[:60]))
    reply, _, _ = chat("List the first five prime numbers, comma-separated.")
    check("primes", all(p in reply for p in ("2", "3", "5", "7", "11")),
          repr(reply[:80]))
    reply, _, _ = chat('Repeat this exactly, nothing else: BANANA-42')
    check("echo", "BANANA-42" in reply, repr(reply[:60]))
else:
    reply, usage, _ = chat("Hello", max_tokens=16)
    check("generates tokens", usage["completion_tokens"] > 0,
          f"completion_tokens={usage['completion_tokens']}")

# 3a. short prefill (prompt M<=64 -> skinny wmma route)
reply, usage, dt = chat("Say hello.", max_tokens=16)
check("short-prefill (skinny route)", len(reply.strip()) > 0,
      f"prompt_toks={usage['prompt_tokens']} ({dt:.1f}s)")
ok_short_m = usage["prompt_tokens"] <= 64
check("short prompt really <=64 tokens", ok_short_m,
      str(usage["prompt_tokens"]))

# 3b. long prefill (M>64 -> marlin route)
long_prompt = ("The quick brown fox jumps over the lazy dog. " * 120
               + "\nIn one word, what animal jumps in the sentence above?")
reply, usage, dt = chat(long_prompt, max_tokens=16)
long_ok = usage["prompt_tokens"] > 64 and (
    STRUCTURAL or "fox" in reply.lower())
check("long-prefill (marlin route)", long_ok,
      f"prompt_toks={usage['prompt_tokens']} reply={reply.strip()[:40]!r} "
      f"({dt:.1f}s)")

# 4. greedy determinism across runs (kernel/graph stability)
q = "Explain in exactly one sentence why the sky is blue."
r1, _, _ = chat(q, max_tokens=64)
r2, _, _ = chat(q, max_tokens=64)
check("greedy determinism", r1 == r2,
      "identical" if r1 == r2 else f"{r1[:40]!r} != {r2[:40]!r}")

# 5. concurrency = max_num_seqs
with cf.ThreadPoolExecutor(2) as ex:
    futs = [ex.submit(chat, f"Count from {i} to {i+4}, comma-separated.", 32)
            for i in (1, 6)]
    outs = [f.result() for f in futs]
check("2-way concurrency", all(len(o[0].strip()) > 0 for o in outs),
      f"{outs[0][0].strip()[:24]!r} | {outs[1][0].strip()[:24]!r}")

# 6. decode TPOT probe: same prompt, 8 vs 128 new tokens
prompt = "Write a long story about a lighthouse keeper."
_, u8, t8 = chat(prompt, max_tokens=8)
_, u128, t128 = chat(prompt, max_tokens=128)
gen = u128["completion_tokens"] - u8["completion_tokens"]
tpot = (t128 - t8) / max(gen, 1) * 1000
check("decode TPOT probe", 0 < tpot < 2000,
      f"{tpot:.0f} ms/token ({1000/tpot:.1f} tok/s decode, TP4)")

print()
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"E2E PIPELINE: {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'} "
      f"({len(results) - n_fail}/{len(results)} checks)")
raise SystemExit(1 if n_fail else 0)
