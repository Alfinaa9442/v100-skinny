"""Run ninfer's EXACT long_decode_aime26 fixtures (verbatim messages
from their repo) against our live server. Usage:
  aime_exact.py <k> <greedy|ninfer>
ninfer sampling profile: temp 0.6, top-p 0.95, top-k 20, presence 1.0.
Output columns match their performance.md conventions. max_new capped to
fit our context (their 65536 exceeds it; per their own method note a
budget-capped run remains a valid sustained-decode sample)."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
import os
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import kernel_src, out_csv, fixtures_dir  # noqa: E402
MODEL = os.environ.get("PROBE_MODEL", "qwen3.6-27b-nvfp4-skinny")
K = int(sys.argv[1])
MODE = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else None
MAXTOK = int(os.environ.get("AIME_MAX_TOKENS", "20000"))
FIXDIR = fixtures_dir()

FIXTURES = ["long_decode_aime26_01", "long_decode_aime26_15",
            "long_decode_aime26_30"]
_fix_sel = os.environ.get("AIME_FIXTURE", "")
if _fix_sel:
    FIXTURES = [f for f in FIXTURES if _fix_sel in f]


def met():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {"d": 0.0, "a": 0.0, "s": 0.0, "pp": {}}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["d"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["a"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            out["s"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            head, val = line.rsplit(" ", 1)
            pos = head.split('position="')[1].split('"')[0]
            out["pp"][pos] = out["pp"].get(pos, 0.0) + float(val)
    return out


def stream(messages):
    body = {
        "model": MODEL, "messages": messages,
        "max_completion_tokens": MAXTOK,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True, "stream_options": {"include_usage": True},
    }
    if MODE == "greedy":
        body["temperature"] = 0
    elif MODE == "tight":
        # flat-phase-equivalent of ninfer's within-candidate top-p
        body.update({"temperature": 0.6, "top_p": 0.7, "top_k": 20,
                     "presence_penalty": 1.0})
    else:
        body.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "presence_penalty": 1.0})
    if SEED is not None:
        body["seed"] = SEED
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    tf = tl = None
    gen = 0
    text_tail = ""
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            t = time.perf_counter()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                gen = d["usage"]["completion_tokens"]
                continue
            ch = d.get("choices")
            if ch:
                delta = (ch[0]["delta"].get("content") or
                         ch[0]["delta"].get("reasoning"))
                if delta:
                    if tf is None:
                        tf = t
                    tl = t
                    text_tail = (text_tail + delta)[-120:]
    return gen, tf, tl, text_tail


print(f"mode={MODE} k={K} maxtok={MAXTOK}")
print(f"{'fixture':>22} {'gen_tok':>8} {'decode tok/s':>12} {'acc%':>6} "
      f"{'tok/round':>9} {'pos1':>5} {'pos2':>5} {'pos3':>5} {'3pos':>6}")
for name in FIXTURES:
    with open(f"{FIXDIR}/{name}.json") as f:
        messages = json.load(f)
    m0 = met()
    gen, tf, tl, tail = stream(messages)
    m1 = met()
    steps = m1["s"] - m0["s"]
    acc = m1["a"] - m0["a"]
    drafts = m1["d"] - m0["d"]
    tau = acc / steps if steps else 0.0
    span = tl - tf if tf and tl else 0.0
    dtoks = (gen - (tau + 1)) / span if span > 0 else 0.0
    p = {i: (m1["pp"].get(str(i), 0) - m0["pp"].get(str(i), 0)) / steps * 100
         for i in (1, 2, 3)}
    boxed = "\\boxed" in tail or "boxed{" in tail
    print(f"{name:>22} {gen:>8} {dtoks:>12.1f} "
          f"{acc/max(drafts,1)*100:>6.1f} {tau+1:>9.2f} "
          f"{p[1]:>5.0f} {p[2]:>5.0f} {p[3]:>5.0f} "
          f"{(p[1]+p[2]+p[3])/3:>6.1f}  tail_boxed={boxed}", flush=True)
print(f"AIME_EXACT_DONE k={K} mode={MODE} seed={SEED}")
