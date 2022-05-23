from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union, Type

import numpy as np
import numpy.typing as npt

from tt_sketch.tensor import SparseTensor
from tt_sketch.utils import ArrayGenerator, ArrayList
from tt_sketch.drm_base import DRM
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import CansketchSparse
# from tt_sketch.drm.orthog_tt_drm import OrthogTTDRM, orth_step


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


# def sparse_sketch(
#     tensor: SparseTensor,
#     left_drm: CansketchSparse,
#     right_drm: CansketchSparse,
#     orthogonalize: bool = False,
# ) -> SketchContainer:
#     """Perform TT-sketch on sparse tensor.
#     Returns the Y-cores and Z-matrices for this sketch."""
#     n_dims = len(tensor.shape)
#     left_rank = left_drm.rank
#     right_rank = right_drm.rank[::-1]
#     Y_mats = list(left_drm.sketch_sparse(tensor))
#     X_mats = list(right_drm.sketch_sparse(tensor))

#     Psi_cores: ArrayList = []
#     Omega_mats: ArrayList = []

#     # Compute Omega matrices
#     for mu in range(n_dims - 1):
#         T_mu = tensor.entries
#         Z_mat = (Y_mats[mu] * T_mu) @ X_mats[mu].T
#         Omega_mats.append(Z_mat)

#     if orthogonalize:
#         left_psi_drm = OrthogTTDRM(left_drm.rank, tensor, "sketch_sparse")
#     # Compute Psi cores
#     for mu in range(n_dims):
#         if mu == 0:
#             left_sketch_vec = None
#             r1 = 1
#         else:
#             if orthogonalize:
#                 left_psi_drm.add_core(Psi_cores[-1])
#                 left_sketch_vec = next(left_psi_drm)
#             else:
#                 left_sketch_vec = Y_mats[mu - 1]
#             r1 = left_rank[mu - 1]

#         if mu == n_dims - 1:
#             right_sketch_vec = None
#             r2 = 1
#         else:
#             right_sketch_vec = X_mats[mu]
#             r2 = right_rank[mu]

#         n_mu = tensor.shape[mu]
#         Psi = np.zeros((r1, n_mu, r2))
#         for j in range(n_mu):
#             Psi[:, j, :] = _Psi_core_slice(
#                 tensor.indices,
#                 tensor.entries,
#                 left_sketch_vec,
#                 right_sketch_vec,
#                 mu,
#                 j,
#             )
#         if orthogonalize and mu < n_dims - 1:
#             Psi = orth_step(Psi, Omega_mats[mu])
#         Psi_cores.append(Psi)

#     return SketchContainer(Psi_cores, Omega_mats)


def sketch_omega_sparse(
    left_sketch: npt.NDArray,
    right_sketch: npt.NDArray,
    *,
    tensor: SparseTensor,
    **kwargs,
) -> npt.NDArray:
    return (left_sketch * tensor.entries) @ right_sketch.T


def sketch_psi_sparse(
    left_sketch: npt.NDArray,
    right_sketch: npt.NDArray,
    *,
    tensor: SparseTensor,
    mu: int,
    psi_shape: Tuple[int, int, int],
    **kwargs,
) -> npt.NDArray:
    Psi = np.zeros(psi_shape)
    n_mu = psi_shape[1]
    for j in range(n_mu):
        Psi[:, j, :] = _Psi_core_slice(
            tensor.indices,
            tensor.entries,
            left_sketch,
            right_sketch,
            mu,
            j,
        )
    return Psi
