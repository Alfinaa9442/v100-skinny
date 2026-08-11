"""Decode-only re-probe against the LIVE server (no reboot).

Streams each cell; t_first = first content chunk (end of prefill +
round 1), t_last = last chunk. Decode-only figures exclude prefill and
round 1:
  decode tok/s  = (gen - (tau+1)) / (t_last - t_first)
  decode ms/rnd = (t_last - t_first) / (steps - 1)
tau/steps from the same metrics deltas as k_sweep_probe. Appends rows
tagged +qpnD to k_sweep_matrix.csv (gen column = decode-span tokens).
"""
import json
import sys
import time
import urllib.request

import os as _os

BASE = "http://127.0.0.1:8000"
MODEL = _os.environ.get("PROBE_MODEL", "qwen3.6-27b-nvfp4-skinny")
K = int(sys.argv[1])
OUT = "/home/user/flatness-run/k_sweep_matrix.csv"

EXTRACT_PASSAGE = ("The lighthouse keeper checked the lamp at dusk. Salt wind "
                   "pressed against the tower glass. Ships passed far out. ") * 30
MATH = ("A train leaves station A at 60 km/h. Two hours later a second "
        "train leaves the same station at 90 km/h on a parallel track. "
        "How far from the station does the second train catch the first? "
        "Work through this carefully.")
CODE = ("Write a complete Python implementation of a red-black tree with "
        "insert, delete, and search, with docstrings.")
PROSE = ("Write a long, detailed short story about a lighthouse keeper "
         "who discovers something unusual washed ashore. Keep going.")
# Cell sets: default = the frozen 6-cell campaign suite. PROBE_CELLS=
# thinkext selects the thinking-mode pairs (defined after CELLS).
CELLS = [
    ("prose-nothink-1024", PROSE, False, 1024),
    ("math-think-2048", MATH, True, 2048),
    ("code-nothink-1024", CODE, False, 1024),
    ("struct-json-1024",
     "Generate a JSON array of 40 user records with fields id (sequential "
     "integer), name, email, signup_date (ISO 8601), and plan (one of "
     "free/pro/enterprise). Output only the JSON.", False, 1024),
    ("struct-csv-1024",
     "Produce a CSV table with 50 rows and columns product_id, name, "
     "category, price, stock. Realistic hardware-store products. Output "
     "only the CSV.", False, 1024),
    ("extract-nothink-512",
     "Repeat the following passage exactly:\n\n" + EXTRACT_PASSAGE,
     False, 512),
]

JSONP = ("Generate a JSON array of 40 user records with fields id "
         "(sequential integer), name, email, signup_date (ISO 8601), and "
         "plan (one of free/pro/enterprise). Output only the JSON.")
CSVP = ("Produce a CSV table with 50 rows and columns product_id, name, "
        "category, price, stock. Realistic hardware-store products. Output "
        "only the CSV.")
EXTRACTP = "Repeat the following passage exactly:\n\n" + EXTRACT_PASSAGE
THINKEXT_CELLS = [
    ("prose-think-2048", PROSE, True, 2048),
    ("code-think-2048", CODE, True, 2048),
    ("struct-json-think-2048", JSONP, True, 2048),
    ("math-nothink-1024", MATH, False, 1024),
]
# PROBE_CELLS=full: every domain in BOTH thinking modes — the complete
# per-category think/nothink matrix for a serving profile.
FULL_CELLS = [
    ("prose-think-2048", PROSE, True, 2048),
    ("prose-nothink-1024", PROSE, False, 1024),
    ("math-think-2048", MATH, True, 2048),
    ("math-nothink-1024", MATH, False, 1024),
    ("code-think-2048", CODE, True, 2048),
    ("code-nothink-1024", CODE, False, 1024),
    ("json-think-2048", JSONP, True, 2048),
    ("json-nothink-1024", JSONP, False, 1024),
    ("csv-think-2048", CSVP, True, 2048),
    ("csv-nothink-1024", CSVP, False, 1024),
    ("extract-think-1024", EXTRACTP, True, 1024),
    ("extract-nothink-512", EXTRACTP, False, 512),
]
_CELL_SETS = {"thinkext": THINKEXT_CELLS, "full": FULL_CELLS}
CELLS = _CELL_SETS.get(_os.environ.get("PROBE_CELLS", ""), CELLS)


def met():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    a = s = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            s += float(line.rsplit(" ", 1)[1])
    return a, s


def stream(prompt, thinking, mt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_completion_tokens": mt,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    t0 = time.perf_counter()
    tf = tl = None
    gen = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
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
            if ch and (ch[0]["delta"].get("content")
                       or ch[0]["delta"].get("reasoning")):
                if tf is None:
                    tf = t
                tl = t
    return gen, t0, tf, tl


rows = []
for name, prompt, thinking, mt in CELLS:
    a0, s0 = met()
    gen, t0, tf, tl = stream(prompt, thinking, mt)
    a1, s1 = met()
    acc, steps = a1 - a0, s1 - s0
    tau = acc / steps if steps else 0.0
    span = tl - tf
    dtoks = (gen - (tau + 1)) / span if span > 0 else 0.0
    dms = span / (steps - 1) * 1000 if steps > 1 else 0.0
    ttft = tf - t0
    pred = (tau + 1) * 1000.0 / dms if dms > 0 else 0.0
    chk_err = abs(dtoks - pred) / dtoks * 100 if dtoks else 0.0
    chk = "OK" if (K == 0 or chk_err <= 5) else "IDENTITY_FAIL"
    print(f"k={K} {name}: DECODE {dtoks:.1f} tok/s  tau={tau:.2f}  "
          f"{dms:.1f} ms/round  rounds={steps:.0f}  gen={gen}  "
          f"ttft={ttft*1000:.0f} ms  (incl-prefill {gen/(tl-t0):.1f})  CHECK[{chk} pred={pred:.1f} err={chk_err:.1f}%]")
    rows.append(f"{time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())},"
                f"{K},{name}+qpnD,{gen},{dtoks:.1f},-,{tau:.2f},{dms:.1f},")

with open(OUT, "a") as f:
    f.write("\n".join(rows) + "\n")
print("decode-only rows appended")
