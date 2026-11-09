# NCCL memory for vLLM's single-rank process groups

vLLM creates the world group and one group per parallel dimension (TP, PP, DP, PCP, EP, ETP), and each is an NCCL
communicator even when the job is one process. With NCCL's default channel count each communicator holds hundreds of
MiB of buffers. On a 24 GiB card that is the difference between the NVFP4 30B acceptance stand-in fitting or not.

## Reproduce

From an empty directory, with the tuned build venv's python (any torch with a CUDA NCCL works; the numbers below are
for this build's torch):

```
python -I nccl_single_rank_memory.py                       # NCCL defaults
NCCL_MAX_NCHANNELS=2 python -I nccl_single_rank_memory.py  # 2 channels
```

It creates the world group plus 6 single-rank subgroups, runs one `all_reduce` in each (so the communicator really
exists), and prints free GPU memory (`torch.cuda.mem_get_info`) after each.

## Observed (2026-11-09)

RTX 4090 24 GiB under WSL2 (Ubuntu 24.04, Windows driver 621.14), torch `2.15.0+rtx40.cu134.20261109.0` with its
statically linked NCCL 2.32.3 (`v0.20261107.0-rtx40-cu134`):

| step | default channels | NCCL_MAX_NCHANNELS=2 |
|---|---|---|
| CUDA context | 22.45 GiB free | 22.45 GiB free |
| world group | -0.51 GiB | -0.14 GiB |
| each subgroup | -0.38 GiB | -0.02 GiB |
| world + 6 subgroups | -2.78 GiB | -0.24 GiB |

In vLLM itself the engine saw 19.27 GiB free at startup with the defaults (world + 7 groups predicts 19.29) and
22.18 GiB with 2 channels.

Other settings tried, world + 6 subgroups: `NCCL_BUFFSIZE=1048576`, `NCCL_NVLS_ENABLE=0` and `NCCL_RUNTIME_CONNECT=1`
gave -2.78 GiB (no change); `NCCL_CUMEM_ENABLE=0` gave -3.63 GiB (worse).

Not measured: a stock (PyPI) NCCL build, native Linux, other GPUs, and any multi-rank job. A single-rank communicator
moves no data between GPUs, so its channel count should not matter for speed (reasoning, not a measurement).

## Where it is used

- `tuned/gb10_acceptance_test.sh` exports `NCCL_MAX_NCHANNELS=2` for rtx40.
- An rtx40 wheel sets `NCCL_MAX_NCHANNELS=2` itself when the job is one rank across DP and the variable is unset
  (`TunedFlags.NCCL_SINGLE_RANK_FEW_CHANNELS`, set by `tuned/package.sh`, applied in `vllm/v1/worker/gpu_worker.py`).
