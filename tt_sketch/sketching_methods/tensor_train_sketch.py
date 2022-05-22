from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from tt_sketch.drm_base import DRM
from tt_sketch.tensor import TensorTrain
from tt_sketch.utils import ArrayGenerator, ArrayList
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import CansketchTT


# TODO: add option to only sketch specific cores for parallelization
def tensor_train_sketch(
    tensor: TensorTrain,
    left_drm: CansketchTT,
    right_drm: CansketchTT,
    orthogonalize: bool = False,
) -> SketchContainer:
    n_dims = len(tensor.shape)
    left_contractions = list(left_drm.sketch_tt(tensor))
    right_contractions = list(right_drm.sketch_tt(tensor))

    Psi_cores = []
    Omega_mats = []

    # Compute Omega matrices
    for mu in range(n_dims - 1):
        L = left_contractions[mu]
        R = right_contractions[mu]
        Omega_mats.append(L.T @ R)

    # Compute Psi cores
    for mu in range(n_dims):
        tt_core = tensor.cores[mu]
        if mu > 0:
            if not orthogonalize:
                L = left_contractions[mu - 1]
        if mu == 0:
            R = right_contractions[0]
            Y = np.einsum("ijk,kl->ijl", tt_core, R)
        elif mu == n_dims - 1:
            # L = left_contractions[n_dims - 2]
            Y = np.einsum("ij,jkl->ikl", L.T, tt_core)
        else:
            # L = left_contractions[mu - 1]
            R = right_contractions[mu]
            Y = np.einsum("ij,jkl,lm->ikm", L.T, tt_core, R)
        Psi_cores.append(Y)

    return SketchContainer(Psi_cores, Omega_mats)
