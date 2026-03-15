import logging
import math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from torch.utils._typing_utils import not_none

from . import default_hooks as default


__all__ = ["arc_topK_hook"]
logger = logging.getLogger(__name__)

class ArcTopKState:
    r"""

    """
    __slots__ = [
        "process_group",
        "priority_rank",
        "compression_ratio",
        "error_dict",
    ]

    def __init__(
        self,
        process_group,
        priority_rank,
        compression_ratio,
    ):
        logger.info(
            "ArcTopKState: priority_rank=%d, compression_ratio=%.4f; ",
            priority_rank,
            compression_ratio,
        )

        self.process_group = process_group
        self.priority_rank = priority_rank
        self.compression_ratio = compression_ratio

    def __getstate__(self):
        r"""
        Return a ``Dict[str, Any]`` which will be pickled and saved.

        ``process_group`` is not serializable and excluded from
        """
        logger.warning(
            "NOTE: Process group is not serializable and excluded from a saved state."
        )
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot != "process_group"
        }

    def __setstate__(self, state):
        r"""
        Take a provided ``state`` and set to this ``PowerSGDState`` instance.

        ``process_group`` is set to default.
        """
        self.process_group = distributed_c10d._get_default_group()
        logger.warning(
            "NOTE: Process group will be set to a default group (i.e. the world size).\
                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded."
        )
        for slot, value in state.items():
            setattr(self, slot, value)


def arc_topK_hook(
    state: ArcTopKState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    r"""
    Implement ARC-Top-K algorithm.

    This DDP communication hook implements ARC-Top-K gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/2510.26709>`_.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:
    1. calculate global priority

        1.1. let n * m = d, reshape grad of size d into n * m matrix G_i for node i.

        1.2. let V = torch.nrand(n, r) be an n * r matrix, where vec(V) ~ N(0, I_nr) and
        r stands for projection_rank.

        1.3. let priority matrix P_i of node i be matmul(G, V)/sqrt(r), which stands for
        the priority of each row of G in node i.

    2. calculate Indices

        2.1. perform an All-Reduce on P_i to get global priority matrix P = mean(P_i).

        2.2. let Indices be Top-K(diag(matmul(P, P.T))) where K stands for compression_ratio.

    3. gradient compression and All-Reduce

        3.1. let compressed gradient matrix G_i be G[Indices, :] and perform an All-Reduce on G'.

        3.2.

    Args:

    :param state:
    :param bucket:
    :return:
    """

    process_group = state.process_group
    group_to_use = (
        process_group if process_group is not None else not_none(dist.group.WORLD)
    )
    world_size = group_to_use.size()

    def _cal_max_factor(size: int):
        factor: int = 1
        while size % factor == 0 and size / factor > factor:
            factor *= 2
        return factor

    gradient = bucket.buffer()
    d = gradient.numel()
    row_num = _cal_max_factor(d)
    gradient = gradient.view(row_num, -1)

    # calculate global priority
    V = torch.randn(row_num, state.priority_rank, device=gradient.device, dtype=gradient.dtype)
    P = torch.matmul(gradient, V) / math.sqrt(state.priority_rank)
    fut = dist.all_reduce(
        P, group=state.process_group, async_op=True
    ).get_future()

    def compress_and_allreduce(fut):
        score = torch.diag(torch.matmul(P, P.T))
        indices = torch.topk(score, k=state.compression_ratio).indices

        comm_gradient = gradient[indices, :]
        comm_fut = dist.all_reduce(
            comm_gradient, group=state.process_group, async_op=True
        ).get_future()

        def decompress_and_finalize(fut):
            avg_gradient = fut.wait()[0] / world_size
            gradient.zero_()
            gradient[indices, :] = avg_gradient
            return gradient

        return comm_fut.then(decompress_and_finalize)


    return fut.then(compress_and_allreduce).then(decompress_and_finalize)
