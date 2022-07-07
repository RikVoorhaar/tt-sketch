from typing import Optional

import numpy as np
import numpy.typing as npt
from tt_sketch.tensor import TuckerTensor
from tt_sketch.utils import matricize


def sketch_omega_tucker(
    left_sketch: npt.NDArray,
    right_sketch: npt.NDArray,
    *,
    tensor: TuckerTensor,
    mu: int,
    **kwargs
):
    mode = tuple(range(mu + 1))
    core_mat = matricize(tensor.core, mode, mat_shape=True)
    return left_sketch.T @ core_mat @ right_sketch


def sketch_psi_tucker(
    left_sketch: Optional[npt.NDArray],
    right_sketch: Optional[npt.NDArray],
    *,
    tensor: TuckerTensor,
    mu: int,
    **kwargs
):
    left_dim = left_sketch.shape[0] if left_sketch is not None else 1
    right_dim = right_sketch.shape[0] if right_sketch is not None else 1
    core_ord3 = tensor.core.reshape(left_dim, tensor.rank[mu], right_dim)
    if left_sketch is None:
        Psi = np.einsum("ijk,kl->ijl", core_ord3, right_sketch)  # type: ignore
    elif right_sketch is None:
        Psi = np.einsum("ij,jkl->ikl", left_sketch.T, core_ord3)
    else:
        Psi = np.einsum(
            "ij,jkl,lm->ikm",
            left_sketch.T,
            core_ord3,
            right_sketch,
            optimize="optimal",
        )
    Psi = np.einsum("ijk,jl->ilk", Psi, tensor.factors[mu])
    return Psi
