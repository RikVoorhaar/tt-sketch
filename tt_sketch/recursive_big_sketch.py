"""Implement's the original, non-streaming, version of the BIGSKETCH algorithm.
This is the dense version, but we can also implement an efficient sparse
version."""

import numpy.typing as npt
import numpy as np
from tt_sketch.utils import (
    ArrayList,
    TTRank,
    process_tt_rank,
    right_mul_pinv,
    matricize,
)


# TODO: Rewrite this to work with sparse / tt / dense sketches
def recursive_big_sketch(
    X: npt.NDArray,
    left_rank: TTRank,
    right_rank: TTRank,
) -> ArrayList:
    shape = X.shape
    n_dims = len(shape)
    left_rank = process_tt_rank(left_rank, shape, trim=True)
    # right rank is allowed to be bigger than TT rank
    right_rank = process_tt_rank(right_rank, shape, trim=False)

    omega_shapes = [
        (np.prod(shape[mu + 1 :]), r) for mu, r in enumerate(right_rank)
    ]
    omega_list = [np.random.normal(size=s) for s in omega_shapes]

    phi_shapes = [
        (np.prod(shape[: mu + 1]), r) for mu, r, in enumerate(left_rank)
    ]
    phi_list = [np.random.normal(size=s) for s in phi_shapes]

    cores = []
    for mu in range(n_dims - 1):
        X_mat = matricize(X, range(mu + 1), mat_shape=True)
        V = X_mat @ omega_list[mu]
        M = phi_list[mu].T @ V
        V_tilde = right_mul_pinv(V, M)

        if mu == 0:
            C, _ = np.linalg.qr(V_tilde)
            core_contracted = C
            cores.append(C.reshape(1, shape[0], left_rank[0]))
        else:
            L = np.kron(core_contracted, np.eye(shape[mu]))
            C_tilde = L.T @ V_tilde
            C, _ = np.linalg.qr(C_tilde)
            core_contracted = L @ C
            cores.append(C.reshape(left_rank[mu - 1], shape[mu], left_rank[mu]))

    X_mat = matricize(X, range(n_dims - 1), mat_shape=True)
    C = core_contracted.T @ X_mat
    cores.append(C.reshape(left_rank[n_dims - 2], shape[n_dims - 1], 1))

    return cores
