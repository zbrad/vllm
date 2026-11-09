"""Measure free GPU memory as world_size=1 NCCL process groups are added, one collective each (as vLLM's groups)."""

import logging
from os import environ

LOGGER = logging.getLogger("nccl_single_rank_memory")


class nccl_single_rank_memory:
    """Creates the world group plus N single-rank NCCL subgroups and reports free memory after each."""

    @staticmethod
    def _free() -> float:
        import torch

        return torch.cuda.mem_get_info()[0] / 2**30

    @staticmethod
    def main() -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(0)
        torch.zeros(1, device="cuda")
        start = nccl_single_rank_memory._free()
        LOGGER.info("CUDA context: free %.2f GiB", start)
        environ.setdefault("MASTER_ADDR", "127.0.0.1")
        environ.setdefault("MASTER_PORT", "29534")
        dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda:0"))
        t = torch.ones(1024, device="cuda")
        dist.all_reduce(t)
        torch.cuda.synchronize()
        last = nccl_single_rank_memory._free()
        LOGGER.info("world group + all_reduce: free %.2f GiB (-%.2f)", last, start - last)
        for n in range(1, 7):
            group = dist.new_group(ranks=[0], backend="nccl")
            dist.all_reduce(t, group=group)
            torch.cuda.synchronize()
            now = nccl_single_rank_memory._free()
            LOGGER.info("subgroup %d + all_reduce: free %.2f GiB (-%.2f, total -%.2f)", n, now, last - now, start - now)
            last = now
        dist.destroy_process_group()


if __name__ == "__main__":
    nccl_single_rank_memory.main()
