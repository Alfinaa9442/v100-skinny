#!/usr/bin/env python3
"""Acceptance probe: per-segment acceptance vs generation length on
natural text, plus extraction control. Greedy decode throughout."""
import json
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.6-27b-nvfp4-skinny"


def met():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        t = r.read().decode()
    g = lambda k: sum(float(v) for v in re.findall(
        rf"vllm:{k}\S*\s+([0-9.e+]+)", t)) or 0.0
    return (g("spec_decode_num_draft_tokens_total"),
            g("spec_decode_num_accepted_tokens_total"),
            g("iteration_tokens_total_count"))


def gen(content, mt):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_completion_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    n = d["usage"]["completion_tokens"]
    return n, time.time() - t0


print("segment, gen_tokens, tok/s, steps, tok/step, drafts, acc, acc%")
prompt = ("Write a very long, detailed story about a lighthouse keeper "
          "who discovers something unusual. Keep going as long as you can.")
prev_d, prev_a, prev_s = met()
total = 0
for seg_target in (256, 512, 1024, 2048):
    n, wall = gen(prompt if total == 0 else
                  prompt + " Continue the story further with new events.",
                  seg_target)
    d, a, s = met()
    dd, da, ds = d - prev_d, a - prev_a, s - prev_s
    prev_d, prev_a, prev_s = d, a, s
    total += n
    print(f"natural-{seg_target}, {n}, {n/wall:.1f}, {ds:.0f}, "
          f"{n/max(ds,1):.2f}, {dd:.0f}, {da:.0f}, "
          f"{100*da/max(dd,1):.1f}%")

passage = ("The lighthouse keeper checked the lamp at dusk. Salt wind "
           "pressed against the tower glass. Ships passed far out. ") * 30
n, wall = gen("Repeat the following passage exactly:\n\n" + passage, 512)
d, a, s = met()
dd, da = d - prev_d, a - prev_a
print(f"extraction-512, {n}, {n/wall:.1f}, -, -, {dd:.0f}, {da:.0f}, "
      f"{100*da/max(dd,1):.1f}%")
