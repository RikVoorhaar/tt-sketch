from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union, Type

import numpy as np
import numpy.typing as npt

from tt_sketch.tensor import SparseTensor
from tt_sketch.utils import ArrayList
from tt_sketch.drm_base import DRM
from tt_sketch.sketch_container import SketchContainer


class CansketchSparse(DRM, ABC):
    rank: Tuple[int, ...]

    @abstractmethod
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayList:
        """Computes list of sketching matrices sampled into a vector using the
        indices of ``tensor`` for each unfolding. Shape of each vector is
        ``v[mu] = (rank[mu], tensor.nnz)``. This way the contraction between
        ``tensor`` and the sketching matrix is of form ``np.dot(tensor.entries,
        v[mu])``"""


def _Psi_core_slice(
    inds: npt.NDArray[Union[np.int64, np.uint64]],
    entries: npt.NDArray,
    left_sketch_vec: Optional[npt.NDArray],
    right_sketch_vec: Optional[npt.NDArray],
    mu: int,
    j: int,
) -> npt.NDArray[np.float64]:
    """Compute slice of tensor Y[:,j,:] from left+right sketching matrix
    and entries+index of sparse tensor"""
    mask = inds[mu] == j

    if mu == 0:  # only sketch on right
        if right_sketch_vec is None:
            raise ValueError
        Psi_slice = entries[mask] @ right_sketch_vec[:, mask].T
        Psi_slice = Psi_slice.reshape(1, -1)
    elif mu == inds.shape[0] - 1:  # only sketch on left
        if left_sketch_vec is None:
            raise ValueError
        Psi_slice = left_sketch_vec[:, mask] @ entries[mask]
        Psi_slice = Psi_slice.reshape(-1, 1)
    else:  # sketch on both sides
        if left_sketch_vec is None or right_sketch_vec is None:
            raise ValueError
        Psi_slice = (
            left_sketch_vec[:, mask] * entries[mask]
        ) @ right_sketch_vec[:, mask].T
    return Psi_slice


def sparse_sketch(
    tensor: SparseTensor,
    left_drm: CansketchSparse,
    right_drm: CansketchSparse,
) -> SketchContainer:
    """Perform TT-sketch on sparse tensor.
    Returns the Y-cores and Z-matrices for this sketch."""
    d = len(tensor.shape)
    left_rank = left_drm.rank
    right_rank = right_drm.rank[::-1]
    Y_mats = left_drm.sketch_sparse(tensor)
    X_mats = right_drm.sketch_sparse(tensor)

    Psi_cores = []
    Omega_mats = []

    # Compute Psi cores
    for mu in range(d):
        if mu == 0:
            left_sketch_vec = None
            r1 = 1
        else:
            left_sketch_vec = Y_mats[mu - 1]
            r1 = left_rank[mu - 1]

        if mu == d - 1:
            right_sketch_vec = None
            r2 = 1
        else:
            right_sketch_vec = X_mats[mu]
            r2 = right_rank[mu]

        n_mu = tensor.shape[mu]
        Psi = np.zeros((r1, n_mu, r2))
        for j in range(n_mu):
            Psi[:, j, :] = _Psi_core_slice(
                tensor.indices,
                tensor.entries,
                left_sketch_vec,
                right_sketch_vec,
                mu,
                j,
            )
        Psi_cores.append(Psi)

    # Compute Z matrices
    for mu in range(d - 1):
        T_mu = tensor.entries
        Z_mat = (Y_mats[mu] * T_mu) @ X_mats[mu].T
        Omega_mats.append(Z_mat)

    return SketchContainer(Psi_cores, Omega_mats)
