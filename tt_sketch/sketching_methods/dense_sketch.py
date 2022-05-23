from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt
from tt_sketch.drm_base import DRM
from tt_sketch.tensor import DenseTensor
from tt_sketch.utils import ArrayGenerator, matricize
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import CansketchDense


# def dense_sketch(
#     tensor: DenseTensor,
#     left_drm: CansketchDense,
#     right_drm: CansketchDense,
#     orthogonalize: bool = False,
# ) -> SketchContainer:
#     n_dims = len(tensor.shape)
#     tensor_data = tensor.data
#     Y_mats = list(left_drm.sketch_dense(tensor))
#     X_mats = list(right_drm.sketch_dense(tensor))

#     Psi_cores = []
#     Omega_mats = []

#     for mu in range(n_dims):
#         if mu == 0:
#             tensor_mat = matricize(tensor_data, 0, mat_shape=True)
#             Psi = tensor_mat @ X_mats[0].T
#             Omega = Y_mats[0] @ Psi
#             Psi = Psi.reshape((1,) + Psi.shape)
#             Psi_cores.append(Psi)
#             Omega_mats.append(Omega)
#         elif mu == n_dims - 1:
#             tensor_mat = matricize(tensor_data, n_dims - 1, mat_shape=True).T
#             Psi = Y_mats[mu - 1] @ tensor_mat
#             Psi = Psi.reshape(Psi.shape + (1,))
#             Psi_cores.append(Psi)
#         else:
#             # Compute order 3 tensorisation of X
#             tensor_ord3 = matricize(tensor_data, range(mu + 1), mat_shape=False)
#             tensor_ord3 = tensor_ord3.reshape(
#                 np.prod(tensor_ord3.shape[:mu], dtype=int),
#                 tensor_ord3.shape[mu],
#                 tensor_ord3.shape[mu + 1],
#             )
#             Psi = np.einsum(
#                 "ij,jkl,ml->ikm", Y_mats[mu - 1], tensor_ord3, X_mats[mu]
#             )
#             Psi_cores.append(Psi)

#             tensor_mat = tensor_ord3.reshape(
#                 tensor_ord3.shape[0] * tensor_ord3.shape[1],
#                 tensor_ord3.shape[2],
#             )
#             Omega = Y_mats[mu] @ tensor_mat @ X_mats[mu].T
#             Omega_mats.append(Omega)

#     return SketchContainer(Psi_cores, Omega_mats)


def sketch_omega_dense(
    left_sketch: npt.NDArray,
    rigth_sketch: np.ndarray,
    *,
    tensor: DenseTensor,
    mu: int,
    **kwargs
) -> npt.NDArray:
    X_mat = matricize(tensor.data, range(mu + 1), mat_shape=True)
    return left_sketch @ X_mat @ rigth_sketch.T


def sketch_psi_dense(
    left_sketch: npt.NDArray,
    right_sketch: np.ndarray,
    *,
    tensor: DenseTensor,
    mu: int,
    **kwargs
) -> npt.NDArray:
    ndim = tensor.ndim
    tensor_data = tensor.data
    if left_sketch is None:
        tensor_mat = matricize(tensor_data, 0, mat_shape=True)
        Psi = tensor_mat @ right_sketch.T
        Psi = Psi.reshape((1,) + Psi.shape)
    elif right_sketch is None:
        tensor_mat = matricize(tensor_data, ndim - 1, mat_shape=True).T
        Psi = left_sketch @ tensor_mat
        Psi = Psi.reshape(Psi.shape + (1,))
    else:
        # Compute order 3 tensorisation of X
        tensor_ord3 = matricize(tensor_data, range(mu + 1), mat_shape=False)
        tensor_ord3 = tensor_ord3.reshape(
            np.prod(tensor_ord3.shape[:mu], dtype=int),
            tensor_ord3.shape[mu],
            tensor_ord3.shape[mu + 1],
        )
        Psi = np.einsum(
            "ij,jkl,ml->ikm", left_sketch, tensor_ord3, right_sketch
        )
    return Psi
