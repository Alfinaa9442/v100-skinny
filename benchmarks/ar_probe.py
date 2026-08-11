"""TP4 allreduce latency at speculative-verify message sizes.

Run: torchrun --nproc_per_node=4 ar_probe.py
Message = M x 5120 fp16 (per-layer TP allreduce payload); 2 collectives
per layer x 64 layers = 128 per forward step. Reports eager and
CUDA-graph-replayed chains (serving replays captured graphs, so the
graphed number is what a verify step actually pays), vs the session-5
ledger's ~3 ms/step and the ~8-15 us NVLink one-shot floor.
"""
import os

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

H = 5120
PER_STEP = 128  # 2 allreduces/layer x 64 layers
ITERS = 300


def bench_eager(x):
    for _ in range(50):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        dist.all_reduce(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1000  # us/op


def bench_graphed(x):
    # capture a 32-op chain, replay it; per-op = replay/32
    for _ in range(10):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for _ in range(32):
                dist.all_reduce(x)
    except Exception as ex:
        return float("nan"), f"capture failed: {type(ex).__name__}"
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(30):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 30 / 32 * 1000, ""


if local_rank == 0:
    print(f"{'M':>3} {'KB':>5} {'eager us/op':>12} {'graph us/op':>12} "
          f"{'ms/step(128, graph)':>20}")
for M in (1, 4, 8, 11, 16, 32):
    x = torch.randn(M * H, dtype=torch.float16, device="cuda")
    te = bench_eager(x)
    tg, note = bench_graphed(x)
    if local_rank == 0:
        step = (tg if tg == tg else te) * PER_STEP / 1000
        print(f"{M:>3} {M*H*2/1024:>5.0f} {te:>12.1f} {tg:>12.1f} "
              f"{step:>20.2f}  {note}")
dist.destroy_process_group()
if local_rank == 0:
    print("AR_PROBE_DONE")
