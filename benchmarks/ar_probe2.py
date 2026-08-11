"""TP4 allreduce through vLLM's real serving entry
(tensor_model_parallel_all_reduce) — measures what a verify round
actually pays, eager and CUDA-graph-replayed (vllm graph_capture context
so the custom NVLink one-shot registers, exactly as serving captures).
Compare against raw NCCL (ar_probe.py: ~25 us/op graphed) and the
one-shot floor (~10-15 us). Run: torchrun --nproc_per_node=4 ar_probe2.py
"""
import os

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
)

_cfg_ctx = set_current_vllm_config(VllmConfig())
_cfg_ctx.__enter__()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
init_distributed_environment(
    world_size=int(os.environ["WORLD_SIZE"]),
    rank=int(os.environ["RANK"]),
    local_rank=local_rank,
    distributed_init_method="env://",
)
initialize_model_parallel(tensor_model_parallel_size=4)

H = 5120
PER_STEP = 128
ITERS = 300


def bench_eager(x):
    for _ in range(50):
        x = tensor_model_parallel_all_reduce(x)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        x = tensor_model_parallel_all_reduce(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1000


def bench_graphed(x):
    with graph_capture(device=device) as ctx:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=ctx.stream):
            y = x
            for _ in range(32):
                y = tensor_model_parallel_all_reduce(y)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(30):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 30 / 32 * 1000


if local_rank == 0:
    print(f"{'M':>3} {'KB':>5} {'eager us/op':>12} {'graph us/op':>12} "
          f"{'ms/step(128)':>13}")
for M in (1, 4, 8, 11, 16):
    x = torch.randn(M * H, dtype=torch.float16, device=device)
    te = bench_eager(x.clone())
    tg = bench_graphed(x.clone())
    if local_rank == 0:
        print(f"{M:>3} {M*H*2/1024:>5.0f} {te:>12.1f} {tg:>12.1f} "
              f"{tg*PER_STEP/1000:>13.2f}", flush=True)
if local_rank == 0:
    print("AR_PROBE2_DONE")
