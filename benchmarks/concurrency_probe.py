"""Concurrency probe: aggregate + per-stream decode throughput at N
simultaneous streams. Usage: concurrency_probe.py <levels> <workload>
e.g. concurrency_probe.py 1,2,4,8 natural
Appends rows to ~/flatness-run/concurrency_matrix.csv.
"""
import json
import sys
import time
import threading
import urllib.request

import os as _os

BASE = "http://127.0.0.1:8000"
MODEL = _os.environ.get("PROBE_MODEL", "qwen3.6-27b-nvfp4-skinny")
LEVELS = [int(x) for x in sys.argv[1].split(",")]
WORKLOAD = sys.argv[2] if len(sys.argv) > 2 else "natural"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
OUT = "/home/user/flatness-run/concurrency_matrix.csv"
MAX_TOKENS = 512

PROMPTS = {
    "natural": "Write a detailed story about a lighthouse keeper in "
               "chapter {i}. Begin with a different scene than usual.",
    "struct": "Generate a JSON array of 40 user records with fields id "
              "(sequential integer starting at {i}00), name, email, "
              "signup_date (ISO 8601), and plan (one of "
              "free/pro/enterprise). Output only the JSON.",
}


def get_steps():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    s = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:iteration_tokens_total_count"):
            s += float(line.rsplit(" ", 1)[1])
    return s


def one_request(i, results):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": PROMPTS[WORKLOAD].format(i=i)}],
        "temperature": 0,
        "max_completion_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    text = d["choices"][0]["message"].get("content") or ""
    is_degen = text.count("1.1.1") > 0 or len(set(text[-100:])) < 6
    results[i] = (d["usage"]["completion_tokens"], is_degen)


rows = []
for n in LEVELS:
    # warmup round at this concurrency
    results = {}
    threads = [threading.Thread(target=one_request, args=(i, results))
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # timed round
    results = {}
    s0 = get_steps()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=one_request, args=(i + 100, results))
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    s1 = get_steps()
    toks = sum(v[0] for v in results.values())
    n_degen = sum(1 for v in results.values() if v[1])
    agg = toks / wall
    steps = s1 - s0
    ms_step = wall / steps * 1000 if steps else 0.0
    degen_note = f" DEGEN={n_degen}/{n}" if n_degen else ""
    rows.append(f"{time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())},"
                f"{WORKLOAD}{TAG},{n},{toks},{agg:.1f},{agg/n:.1f},"
                f"{ms_step:.1f},{n_degen}")
    print(f"streams={n}: aggregate {agg:.1f} tok/s "
          f"({agg/n:.1f}/stream), ms/step={ms_step:.1f}, "
          f"toks={toks}{degen_note}")

import os
new = not os.path.exists(OUT)
with open(OUT, "a") as f:
    if new:
        f.write("ts,workload,streams,gen_tokens,aggregate_tok_s,"
                "per_stream_tok_s,ms_step\n")
    f.write("\n".join(rows) + "\n")
print("rows appended")
