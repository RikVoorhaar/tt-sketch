from typing import Optional

import numpy as np
import numpy.typing as npt

from tt_sketch.tensor import TensorTrain
from tt_sketch.utils import ArrayList
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import CansketchTT
from tt_sketch.drm.orthog_tt_drm import OrthogTTDRM, orth_step


def tensor_train_sketch(
    tensor: TensorTrain,
    left_drm: CansketchTT,
    right_drm: CansketchTT,
    orthogonalize: bool = False,
) -> SketchContainer:
    n_dims = len(tensor.shape)
    left_contractions = list(left_drm.sketch_tt(tensor))
    right_contractions = list(right_drm.sketch_tt(tensor))

    Psi_cores: ArrayList = []
    Omega_mats: ArrayList = []

    # Compute Omega matrices
    for mu in range(n_dims - 1):
        L = left_contractions[mu]
        R = right_contractions[mu]
        Omega_mats.append(L.T @ R)

    if orthogonalize:
        left_psi_drm = OrthogTTDRM(left_drm.rank, tensor, "sketch_tt")

    # Compute Psi cores
    for mu in range(n_dims):
        tt_core = tensor.cores[mu]
        if mu > 0:
            if orthogonalize:
                left_psi_drm.add_core(Psi_cores[-1])
                L = next(left_psi_drm)
            else:
                L = left_contractions[mu - 1]
        if mu == 0:
            R = right_contractions[0]
            Psi = np.einsum("ijk,kl->ijl", tt_core, R)
        elif mu == n_dims - 1:
            Psi = np.einsum("ij,jkl->ikl", L.T, tt_core)
        else:
            R = right_contractions[mu]
            Psi = np.einsum("ij,jkl,lm->ikm", L.T, tt_core, R)
        if orthogonalize and mu < n_dims - 1:
            Psi = orth_step(Psi, Omega_mats[mu])

        Psi_cores.append(Psi)

    return SketchContainer(Psi_cores, Omega_mats)


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
        Psi = np.einsum("ij,jkl,lm->ikm", left_sketch.T, tt_core, right_sketch)
    return Psi
