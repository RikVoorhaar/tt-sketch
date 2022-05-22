from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from tt_sketch.drm_base import DRM
from tt_sketch.tensor import TensorTrain
from tt_sketch.utils import ArrayList
from tt_sketch.sketch_container import SketchContainer


class CansketchTT(DRM, ABC):
    @abstractmethod
    def sketch_tt(self, tensor: TensorTrain) -> ArrayList:
        r"""List of contractions of form :math:`Y_\mu^\top T_{\leq\mu}` where
        :math:`X_\mu` is the DRM, and :math"`T_{\leq\mu}` the
        contraction of the first :math:`\mu` cores of ``tensor``.

        Returns array of shape ``(tensor.rank[mu], drm.rank[mu])``"""


# TODO: add option to only sketch specific cores for parallelization
def tensor_train_sketch(
    tensor: TensorTrain,
    left_drm: CansketchTT,
    right_drm: CansketchTT,
) -> SketchContainer:
    n_dims = len(tensor.shape)
    left_contractions = list(left_drm.sketch_tt(tensor))
    right_contractions = list(right_drm.sketch_tt(tensor))

    Psi_cores = []
    Omega_mats = []

    # Compute Z matrices
    for mu in range(n_dims - 1):
        L = left_contractions[mu]
        R = right_contractions[mu]
        Omega_mats.append(L.T @ R)

    # Compute Y cores
    for mu in range(n_dims):
        tt_core = tensor.cores[mu]
        if mu == 0:
            R = right_contractions[0]
            Y = np.einsum("ijk,kl->ijl", tt_core, R)
        elif mu == n_dims - 1:
            L = left_contractions[n_dims - 2]
            Y = np.einsum("ij,jkl->ikl", L.T, tt_core)
        else:
            L = left_contractions[mu - 1]
            R = right_contractions[mu]
            Y = np.einsum("ij,jkl,lm->ikm", L.T, tt_core, R)
        Psi_cores.append(Y)

    return SketchContainer(Psi_cores, Omega_mats)
