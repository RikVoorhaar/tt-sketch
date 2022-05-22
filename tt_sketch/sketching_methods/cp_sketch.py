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


def cp_sketch(
    tensor: CPTensor, left_drm: CansketchCP, right_drm: CansketchCP
) -> SketchContainer:
    n_dims = len(tensor.shape)
    left_contractions = list(left_drm.sketch_cp(tensor))
    right_contractions = list(right_drm.sketch_cp(tensor))

    Psi_cores = []
    Omega_mats = []

    # Compute Omega matrices
    for mu in range(n_dims - 1):
        L = left_contractions[mu]
        R = right_contractions[mu]
        Omega_mats.append(L.T @ R)

    # Compute Psi cores
    for mu in range(n_dims):
        cp_core = tensor.cores[mu]
        if mu == 0:
            R = right_contractions[0]
            Psi = np.einsum("ji,il->jl", cp_core, R)
            Psi = Psi.reshape((1,) + Psi.shape)
        elif mu == n_dims - 1:
            L = left_contractions[n_dims - 2]
            Psi = np.einsum("il,kl->ik", L.T, cp_core)
            Psi = Psi.reshape(Psi.shape + (1,))
        else:
            L = left_contractions[mu - 1]
            R = right_contractions[mu]
            Psi = np.einsum("ij,kj,jm->ikm", L.T, cp_core, R)
        Psi_cores.append(Psi)
    sketch = SketchContainer(Psi_cores, Omega_mats)

    return sketch
