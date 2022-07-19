from typing import Optional

import numpy as np
import numpy.typing as npt
from tt_sketch.tensor import TensorTrain


def sketch_omega_tt(
    left_sketch: npt.NDArray, right_sketch: npt.NDArray, **kwargs
):
    return left_sketch.T @ right_sketch


def sketch_psi_tt(
    left_sketch: Optional[npt.NDArray],
    right_sketch: Optional[npt.NDArray],
    *,
    tensor: TensorTrain,
    mu: int,
    **kwargs
):
    tt_core = tensor.cores[mu]
    if left_sketch is None:
        Psi = np.einsum("ijk,kl->ijl", tt_core, right_sketch)  # type: ignore
    elif right_sketch is None:
        Psi = np.einsum("ij,jkl->ikl", left_sketch.T, tt_core)
    else:
        Psi = np.einsum(
            "ij,jkl,lm->ikm",
            left_sketch.T,
            tt_core,
            right_sketch,
            optimize="optimal",
        )
    return Psi
