"""Acceptance-gap probe: same weights, same MTP head — find which
workload/metric reproduces the published 73-80% acceptance.

Runs a labeled prompt matrix (domain x thinking x length), computes
acceptance per cell from /metrics deltas, and reports every plausible
metric definition so the published number can be matched:
  - accepted/drafted (our banked definition)
  - mean accepted length per step (tau) and tau/k
  - per-position acceptance (if the server exposes per-pos counters)
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.6-27b-nvfp4-skinny"
K = 4

MATRIX = [
    ("prose-nothink-256", "Write a story about a lighthouse keeper.",
     False, 256),
    ("prose-nothink-2048", "Write a very long detailed story about a "
     "lighthouse keeper. Keep going, do not stop.", False, 2048),
    ("math-think-2048", "A train leaves station A at 60 km/h. Two hours "
     "later a second train leaves the same station at 90 km/h on a "
     "parallel track. How far from the station does the second train "
     "catch the first? Work through this carefully.", True, 2048),
    ("math-nothink-1024", "A train leaves station A at 60 km/h. Two hours "
     "later a second train leaves the same station at 90 km/h on a "
     "parallel track. How far from the station does the second train "
     "catch the first? Work through this carefully.", False, 1024),
    ("code-nothink-1024", "Write a complete Python implementation of a "
     "red-black tree with insert, delete, and search, with docstrings.",
     False, 1024),
    ("code-think-2048", "Write a complete Python implementation of a "
     "red-black tree with insert, delete, and search, with docstrings.",
     True, 2048),
]


def get_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {"drafts": 0.0, "accepted": 0.0, "steps": 0.0, "per_pos": {}}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["drafts"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            out["steps"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            # label form {...position="N"...} value
            head, val = line.rsplit(" ", 1)
            pos = head.split('position="')[1].split('"')[0] \
                if 'position="' in head else "?"
            out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + float(val)
    return out


def chat(prompt, thinking, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return d["usage"]["completion_tokens"]


print(f"{'cell':>20} {'gen':>5} {'acc/draft':>9} {'tau':>5} "
      f"{'tau/k':>6} {'pos1%':>6} {'pos-profile'}")
for name, prompt, thinking, max_tokens in MATRIX:
    m0 = get_metrics()
    gen = chat(prompt, thinking, max_tokens)
    m1 = get_metrics()
    drafts = m1["drafts"] - m0["drafts"]
    acc = m1["accepted"] - m0["accepted"]
    steps = m1["steps"] - m0["steps"]
    pp = {p: m1["per_pos"].get(p, 0) - m0["per_pos"].get(p, 0)
          for p in m1["per_pos"]}
    ad = acc / drafts if drafts else 0.0
    tau = acc / steps if steps else 0.0            # accepted per step
    pos_keys = sorted(pp, key=lambda x: int(x) if x.isdigit() else 99)
    pos1 = (pp[pos_keys[0]] / steps * 100) if steps and pos_keys else 0.0
    prof = " ".join(f"p{p}:{pp[p]/steps*100:.0f}%" for p in pos_keys) \
        if steps and pos_keys else "n/a"
    print(f"{name:>20} {gen:>5} {ad*100:>8.1f}% {tau:>5.2f} "
          f"{(tau/K)*100:>5.1f}% {pos1:>5.1f}% {prof}")
print("\ndefinitions: acc/draft = accepted/drafted (our banked number); "
      "tau = mean accepted per step; tau/k with k=4; "
      "pos1% = first-draft-position acceptance per step")
