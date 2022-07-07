import numpy as np
import numpy.typing as npt
from tt_sketch.tensor import CPTensor


def sketch_omega_cp(
    left_sketch: npt.NDArray, right_sketch: npt.NDArray, **kwargs
) -> npt.NDArray:
    return left_sketch.T @ right_sketch


def sketch_psi_cp(
    left_sketch: npt.NDArray,
    right_sketch: npt.NDArray,
    *,
    tensor: CPTensor,
    mu: int,
    **kwargs,
) -> npt.NDArray:
    cp_core = tensor.cores[mu]

    if left_sketch is None:
        Psi = np.einsum("ji,il->jl", cp_core, right_sketch)
        Psi = Psi.reshape((1,) + Psi.shape)
    elif right_sketch is None:
        Psi = np.einsum("il,kl->ik", left_sketch.T, cp_core)
        Psi = Psi.reshape(Psi.shape + (1,))
    else:
        Psi = np.einsum(
            "ij,kj,jm->ikm",
            left_sketch.T,
            cp_core,
            right_sketch,
            optimize="optimal",
        )
    return Psi
