"""Implements TT-SVD algorithm"""
from typing import Optional

import numpy as np

from tt_sketch.tensor import Tensor, TensorTrain
from tt_sketch.utils import TTRank, matricize, process_tt_rank


def tt_svd(tensor: Tensor, rank: Optional[TTRank] = None) -> TensorTrain:
    """Compute the TT-SVD of an array in a left-to-right sweep"""
    tensor_numpy = tensor.to_numpy()
    shape = tensor.shape
    n_dims = len(shape)
    if rank is None:
        rank = (np.prod(shape, dtype=int),) * (n_dims - 1)
    rank = process_tt_rank(rank, shape, trim=True)
    new_rank = list(rank)
    tt_cores = []

    tensor_mat = matricize(tensor_numpy, 0)
    U, S, V = np.linalg.svd(tensor_mat, full_matrices=False)
    r = max(min(U.shape[1], new_rank[0]), 1)
    new_rank[0] = r
    tt_core = U[:, :r]
    tt_core = tt_core.reshape(1, shape[0], r)
    tt_cores.append(tt_core)

    tensor_compressed = (np.diag(S[:r]) @ V[:r, :]).reshape(
        (r,) + tensor.shape[1:]
    )
    for mu in range(1, n_dims - 1):
        tensor_mat = matricize(tensor_compressed, (0, 1), mat_shape=True)
        U, S, V = np.linalg.svd(tensor_mat, full_matrices=False)
        r = max(min(U.shape[1], new_rank[mu]), 1)
        new_rank[mu] = r
        tt_core = U[:, :r].reshape(new_rank[mu - 1], shape[mu], r)
        tt_cores.append(tt_core)
        tensor_compressed = np.diag(S[:r]) @ V[:r, :]
        tensor_compressed = tensor_compressed.reshape(
            (r,) + tensor.shape[mu + 1 :]
        )

    tt_core = tensor_compressed.reshape(
        new_rank[n_dims - 2], shape[n_dims - 1], 1
    )
    tt_cores.append(tt_core)

    return TensorTrain(tt_cores)
