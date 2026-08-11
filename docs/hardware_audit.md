# Hardware / System Configuration Audit — the 4x V100 Box

System-configuration audit (2026-08-11) of the serving box — Dell PowerEdge C4130, 4x Tesla V100-SXM2-16GB, dual Xeon E5-2690 v4 (Broadwell), two NUMA nodes — for single-stream, latency-critical tensor-parallel decode.

**Headline:** the findings that mattered were NUMA placement (#1–#3). All four GPUs sit on NUMA node 0, but three of four TP ranks — and 85–91% of their host pages — ran on node 1, crossing the socket interconnect (UPI) on every CUDA ioctl, launch doorbell, and NCCL host-side handshake. Fixed via `numactl --cpunodebind=0 --membind=0` in the launch scripts plus `kernel.numa_balancing=0`: worth ~3–4% round time, and it killed a ±3% cross-boot placement lottery.

Workload under audit: vLLM TP=4 decode — memory-bandwidth-bound, ~30 ms speculative rounds, per-layer NCCL allreduce. Because every layer ends in a collective, any per-rank asymmetry (#1, #2, #4) is paid by all ranks: the round runs at the speed of the slowest rank.

## Findings

Severity: HIGH = measurable steady-state loss · MEDIUM = real but bounded · LOW = marginal/hygiene · INFO = no action.

1. **TP workers not NUMA-pinned** (HIGH — fixed). All four workers had CPU affinity 0–55; three of four ranks executed (and migrated) on node 1 while all GPUs are on node 0, putting every CUDA ioctl and NCCL host handshake across UPI (node distance 21 vs 10) and injecting run-to-run jitter. Fixed: launch scripts prefix the server with `numactl --cpunodebind=0 --membind=0`.
2. **Host memory on the wrong socket** (HIGH — fixed). Per-process `numa_maps` showed ~85–91% of three ranks' host pages resident on node 1 — the same UPI penalty as #1 applied to the resident working set. Fixed at launch by `--membind=0` (pages are placed at allocation time; not correctable for a running process).
3. **Automatic NUMA balancing enabled** (MEDIUM — fixed). `kernel.numa_balancing=1` periodically unmaps pages to provoke hinting faults and migrations — pure tail-latency noise that actively fights #1/#2. Fixed: set to 0 and persisted. Findings #1–#3 together account for the ~3–4% round-time gain and removal of the ±3% cross-boot placement lottery.
4. **Application clocks not pinned to max** (MEDIUM — fixed). Application clocks sat at 1312 MHz vs the 1530 MHz max SM clock — a 14.2% deficit; an idle rank drops to the floor and ramps back after every idle gap, gating the collective. Fixed: app clocks pinned (877,1530) via a systemd unit.
5. **Memory clock** (INFO — no deficit). Memory clock 877 MHz = max on all 4 GPUs; theoretical peak 877 MHz x 4096 bit x 2 / 8 = 898 GB/s. The measured 825 GB/s is 91.9% of theoretical, the normal achievable ratio for HBM2, so 825 GB/s is the correct roofline ceiling; no headroom in clock configuration.
6. **Persistence mode off** (MEDIUM — fixed). `nvidia-persistenced` was active but launched with the distro-default `--no-persistence-mode`, so each fresh process re-initialized the driver (~1–3 s) and clock/ECC settings were dropped when the last client detached — the reason #4 would not stick. Fixed: systemd drop-in removes the flag; persistence mode on.
7. **ECC inconsistent across GPUs** (MEDIUM, correctness — fixed). GPU 3 had ECC disabled. No performance or capacity cost either way (V100 HBM2 has native inline ECC; ECC-on and ECC-off GPUs report identical 16384 MiB), but a memory fault would corrupt tokens silently instead of raising Xid 48. Root cause: a vestigial boot service written for a since-replaced faulty card was still disabling ECC on the healthy replacement. Service removed; ECC now enabled on all four GPUs.
8. **Deep C-states disabled only at runtime** (MEDIUM — fixed). C3/C6 were disabled via sysfs (worst-case wakeup 10 µs instead of C6's 133 µs exit latency), but the setting would silently revert on reboot. Fixed: the C3/C6 disable is now durable via a systemd unit. Re-enabling deeper C6 core parking was also measured directly and rejected: +0.1 ms/round.
9. **`vm.swappiness=60`** (LOW). 0 B of swap in use; the risk is a future page-out of a hot dispatch page inside a 30 ms round. Action: lower to 10 and persist — cheap insurance.
10. **THP at `madvise`** (LOW). `madvise` for both `enabled` and `defrag` is the right default here — the GPUs hold the weights, so host THP touches only the control-path working set. `always` is optional and must be A/B-measured before keeping.
11. **PCIe ASPM policy at BIOS default** (LOW — moot). The platform FADT denies the OS ASPM control, and the links already report ASPM Disabled. No action possible or needed.
12. **Compute mode `Default`** (LOW). A stray process can attach a context to a serving GPU mid-benchmark. Optional on a dedicated box: `EXCLUSIVE_PROCESS` (note: blocks multi-process profiling tools).
13. **Volta is end-of-line on this driver branch** (INFO, availability). The proprietary 580-branch driver is required (the open GPU kernel modules support Turing and newer only), and the CUDA 13.x toolkit dropped `sm_70` — builds must stay on a 12.x toolchain (CUDA 12.8, `TORCH_CUDA_ARCH_LIST=7.0`). Action: hold the driver packages and keep the 12.x toolkit.
14. **Unattended kernel updates + DKMS** (INFO). Nine kernels installed, DKMS driver builds against five; a kernel update whose DKMS rebuild fails takes all four GPUs offline at next boot. Action: hold kernel packages on a measurement box; verify `dkms status` after any kernel change.
15. **Historical Xid errors are application-class, not hardware** (INFO). 8x Xid 31 + 6x Xid 43, all in a single day, all near-null-address MMU faults from since-exited Python processes — the signature of an illegal memory access during custom-kernel development. Zero hardware-class Xids (48, 63/64, 74, 79). No hardware action; escalate only if hardware-class codes ever appear.

## Verified clean

Recorded so they are not re-investigated:

- **NVLink:** fully-connected NV2 mesh (every GPU pair), all 6 links up per GPU at 25.781 GB/s, zero NVLink error counters, P2P OK for all 12 ordered pairs — the per-layer allreduce never falls back to PCIe.
- **Memory health:** zero volatile and aggregate ECC errors across all banks, zero retired pages, no pending page blacklist.
- **Thermals / power:** 42–48 °C against an 87 °C slowdown threshold; 56–74 W of a 300 W limit; `clocks_event_reasons = 0x0` — no throttling of any kind.
- **PCIe:** Gen3 x16 current = max on all four GPUs (the platform ceiling for Broadwell + V100); zero AER errors, zero machine checks.
- **CPU:** `performance` governor on all 56 logical CPUs, turbo enabled; GPU IRQs affine to node-0 CPUs, matching GPU locality.
- **Host resources:** `/dev/shm` 63 GiB, ample locked-memory limits, 0 B swap in use, `vm.zone_reclaim_mode=0`, `vm.max_map_count` ample.
