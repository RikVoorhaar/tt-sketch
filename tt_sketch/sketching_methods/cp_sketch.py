from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from tt_sketch.drm_base import DRM
from tt_sketch.tensor import CPTensor
from tt_sketch.utils import ArrayGenerator, ArrayList
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import CansketchCP
from tt_sketch.drm.orthog_tt_drm import OrthogTTDRM, orth_step


def cp_sketch(
    tensor: CPTensor,
    left_drm: CansketchCP,
    right_drm: CansketchCP,
    orthogonalize: bool = False,
) -> SketchContainer:
    n_dims = len(tensor.shape)
    left_contractions = list(left_drm.sketch_cp(tensor))
    right_contractions = list(right_drm.sketch_cp(tensor))

    Psi_cores: ArrayList = []
    Omega_mats: ArrayList = []

    # Compute Omega matrices
    for mu in range(n_dims - 1):
        L = left_contractions[mu]
        R = right_contractions[mu]
        Omega_mats.append(L.T @ R)

    if orthogonalize:
        left_psi_drm = OrthogTTDRM(left_drm.rank, tensor, "sketch_cp")

    # Compute Psi cores
    for mu in range(n_dims):
        cp_core = tensor.cores[mu]
        if mu > 0:
            if orthogonalize:
                left_psi_drm.add_core(Psi_cores[-1])
                L = next(left_psi_drm)
            else:
                L = left_contractions[mu - 1]
        if mu == 0:
            R = right_contractions[0]
            Psi = np.einsum("ji,il->jl", cp_core, R)
            Psi = Psi.reshape((1,) + Psi.shape)
        elif mu == n_dims - 1:
            Psi = np.einsum("il,kl->ik", L.T, cp_core)
            Psi = Psi.reshape(Psi.shape + (1,))
        else:
            R = right_contractions[mu]
            Psi = np.einsum("ij,kj,jm->ikm", L.T, cp_core, R)
        if orthogonalize and mu < n_dims - 1:
            Psi = orth_step(Psi, Omega_mats[mu])
        Psi_cores.append(Psi)
    sketch = SketchContainer(Psi_cores, Omega_mats)

    return sketch


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
        Psi = np.einsum("ij,kj,jm->ikm", left_sketch.T, cp_core, right_sketch)
    return Psi
