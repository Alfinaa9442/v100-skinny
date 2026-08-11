"""Two-phase decode probe: trace vs payload, one uncapped generation
per domain against the live server.

Streams with thinking enabled; snapshots spec metrics at start, at the
reasoning->content transition, and at end. Reports per phase: tokens
(chunk-counted), tau (metrics delta), decode rate — plus the honest
end-to-end request rate. Chunk counts approximate tokens (vLLM streams
~1 token/chunk); usage.completion_tokens anchors the total.
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
import os
MODEL = os.environ.get("PROBE_MODEL", "qwen3.6-27b-nvfp4-skinny")
BUDGET = 8192


def _sampling_params():
    t = os.environ.get("PROBE_TEMP")
    if t is not None:
        if float(t) == 0:
            return {"temperature": 0}
        return {"temperature": float(t), "top_p": 0.95, "top_k": 20,
                "seed": 1001}
    if os.environ.get("PROBE_SAMPLING", "") == "qwen":
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1001}
    return {"temperature": 0}
CELLS = [
    ("prose", "Write a long, detailed short story about a lighthouse "
              "keeper who discovers something unusual washed ashore. "
              "Keep going."),
    ("math", "A train leaves station A at 60 km/h. Two hours later a "
             "second train leaves the same station at 90 km/h on a "
             "parallel track. How far from the station does the second "
             "train catch the first? Work through this carefully."),
    ("code", "Write a complete Python implementation of a red-black "
             "tree with insert, delete, and search, with docstrings."),
    ("json", "Generate a JSON array of 40 user records with fields id "
             "(sequential integer), name, email, signup_date (ISO "
             "8601), and plan (one of free/pro/enterprise). Output "
             "only the JSON."),
    ("csv", "Produce a CSV table with 50 rows and columns product_id, "
            "name, category, price, stock. Realistic hardware-store "
            "products. Output only the CSV."),
    ("extract", "Repeat the following passage exactly:\n\n" +
     ("The lighthouse keeper checked the lamp at dusk. Salt wind "
      "pressed against the tower glass. Ships passed far out. ") * 30),
]


def met():
    text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    a = s = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            s += float(line.rsplit(" ", 1)[1])
    return a, s


for name, prompt in CELLS:
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        **_sampling_params(),
        "max_completion_tokens": BUDGET,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer EMPTY"})
    a0, s0 = met()
    trace_chunks = pay_chunks = 0
    t_first = t_trans = t_last = None
    a1 = s1 = None
    gen = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
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
            if not ch:
                continue
            delta = ch[0]["delta"]
            if delta.get("reasoning"):
                trace_chunks += 1
                t_first = t_first or t
                t_last = t
            elif delta.get("content"):
                if t_trans is None:
                    t_trans = t
                    a1, s1 = met()
                pay_chunks += 1
                t_first = t_first or t
                t_last = t
    a2, s2 = met()
    if a1 is None:  # never reached payload (all trace or all content)
        a1, s1 = a2, s2
        t_trans = t_last
    tr_steps, tr_acc = (s1 - s0), (a1 - a0)
    pl_steps, pl_acc = (s2 - s1), (a2 - a1)
    tr_tau = tr_acc / tr_steps if tr_steps else 0.0
    pl_tau = pl_acc / pl_steps if pl_steps else 0.0
    tr_span = (t_trans - t_first) if (t_trans and t_first) else 0.0
    pl_span = (t_last - t_trans) if (t_trans and t_last) else 0.0
    tot = trace_chunks + pay_chunks
    scale = gen / tot if tot else 1.0
    tr_tok = trace_chunks * scale
    pl_tok = pay_chunks * scale
    tr_rate = tr_tok / tr_span if tr_span > 0 else 0.0
    pl_rate = pl_tok / pl_span if pl_span > 0 else 0.0
    e2e = gen / (t_last - t_first) if (t_last and t_first and
                                       t_last > t_first) else 0.0
    finished = "natural" if gen < BUDGET else "CAPPED"
    print(f"PHASE {name:8s} gen={gen:5d} ({finished})  "
          f"trace: {tr_tok:5.0f} tok  tau={tr_tau:5.2f}  {tr_rate:6.1f} tok/s | "
          f"payload: {pl_tok:5.0f} tok  tau={pl_tau:5.2f}  {pl_rate:6.1f} tok/s | "
          f"e2e {e2e:6.1f} tok/s", flush=True)
print("PHASE_PROBE_DONE")
