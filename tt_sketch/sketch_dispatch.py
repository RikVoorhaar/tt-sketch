from typing import Callable, Dict, Type
from functools import partial

import numpy as np
import numpy.typing as npt

from tt_sketch.drm import TensorTrainDRM

# from tt_sketch.drm.orthog_tt_drm import OrthogTTDRM, orth_step
from tt_sketch.drm_base import DRM
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchCP,
    CansketchDense,
    CansketchSparse,
    CansketchTT,
)
from tt_sketch.sketching_methods.cp_sketch import (
    cp_sketch,
    sketch_omega_cp,
    sketch_psi_cp,
)
from tt_sketch.sketching_methods.dense_sketch import (
    dense_sketch,
    sketch_omega_dense,
    sketch_psi_dense,
)
from tt_sketch.sketching_methods.sparse_sketch import (
    sketch_omega_sparse,
    sketch_psi_sparse,
    sparse_sketch,
)
from tt_sketch.sketching_methods.tensor_train_sketch import (
    sketch_omega_tt,
    sketch_psi_tt,
    tensor_train_sketch,
)
from tt_sketch.tensor import (
    CPTensor,
    DenseTensor,
    SparseTensor,
    Tensor,
    TensorSum,
    TensorTrain,
)
from tt_sketch.utils import ArrayList, right_mul_pinv

SKETCHING_METHODS = {
    CansketchSparse: sparse_sketch,
    CansketchTT: tensor_train_sketch,
    CansketchDense: dense_sketch,
    CansketchCP: cp_sketch,
}

ABSTRACT_TENSOR_SKETCH_DISPATCH = {
    SparseTensor: CansketchSparse,
    TensorTrain: CansketchTT,
    DenseTensor: CansketchDense,
    CPTensor: CansketchCP,
}

DRM_SKETCH_METHOD_DISPATCH = {
    SparseTensor: "sketch_sparse",
    TensorTrain: "sketch_tt",
    DenseTensor: "sketch_dense",
    CPTensor: "sketch_cp",
}


OMEGA_METHODS = {
    SparseTensor: sketch_omega_sparse,
    TensorTrain: sketch_omega_tt,
    DenseTensor: sketch_omega_dense,
    CPTensor: sketch_omega_cp,
}

PSI_METHODS = {
    SparseTensor: sketch_psi_sparse,
    TensorTrain: sketch_psi_tt,
    DenseTensor: sketch_psi_dense,
    CPTensor: sketch_psi_cp,
}


def sketch_omega_sum(
    left_sketch_array: ArrayList,
    right_sketch_array: ArrayList,
    *,
    tensor: TensorSum,
    omega_shape: tuple[int, int],
    **kwargs,
) -> npt.NDArray:
    omega = np.zeros(omega_shape)
    for summand, left_sketch, right_sketch in zip(
        tensor.tensors, left_sketch_array, right_sketch_array
    ):
        omega_method = OMEGA_METHODS[type(summand)]
        omega += omega_method(
            left_sketch,
            right_sketch,
            tensor=summand,
            omega_shape=omega_shape,
            **kwargs,
        )
    return omega


OMEGA_METHODS[TensorSum] = sketch_omega_sum


def sketch_psi_sum(
    left_sketch_array: ArrayList,
    right_sketch_array: ArrayList,
    *,
    tensor: TensorSum,
    psi_shape: tuple[int, int],
    **kwargs,
) -> npt.NDArray:
    psi = np.zeros(psi_shape)
    if left_sketch_array is None:
        left_sketch_array = (None,) * tensor.num_summands
    if right_sketch_array is None:
        right_sketch_array = (None,) * tensor.num_summands

    for summand, left_sketch, right_sketch in zip(
        tensor.tensors, left_sketch_array, right_sketch_array
    ):
        psi_method = PSI_METHODS[type(summand)]
        psi += psi_method(
            left_sketch,
            right_sketch,
            tensor=summand,
            psi_shape=psi_shape,
            **kwargs,
        )
    return psi


PSI_METHODS[TensorSum] = sketch_psi_sum


def sum_sketch(tensor: TensorSum, *, drm: DRM):
    sketch_generators = []
    for summand in tensor.tensors:
        sketch_generators.append(get_sketch_method(summand, drm)(summand))
    for _ in range(len(tensor.shape) - 1):
        yield tuple(next(gen) for gen in sketch_generators)


def get_sketch_method(tensor: Tensor, drm: DRM) -> Callable:
    if type(tensor) in DRM_SKETCH_METHOD_DISPATCH:
        drm_sketch_method = DRM_SKETCH_METHOD_DISPATCH[type(tensor)]
        return getattr(drm, drm_sketch_method)
    elif isinstance(tensor, TensorSum):
        return partial(sum_sketch, drm=drm)
    else:
        raise ValueError(f"DRM of type {type(drm)} can't sketch {type(tensor)}")


def orth_step(
    Psi: npt.NDArray[np.float64], Omega: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    Psi_shape = Psi.shape
    Psi_mat = Psi.reshape((Psi_shape[0] * Psi_shape[1], Psi_shape[2]))
    Psi_mat = right_mul_pinv(Psi_mat, Omega)
    Psi_mat, _ = np.linalg.qr(Psi_mat)
    Psi = Psi_mat.reshape(Psi_shape[0], Psi_shape[1], Omega.shape[0])
    return Psi


class OrthogTTDRM:
    def __init__(self, rank, tensor):
        self.rank = rank
        self.drm = TensorTrainDRM(rank, tensor.shape, transpose=False, cores=[])
        self.generator = None
        self.tensor = tensor
        self.sketch_method = get_sketch_method(tensor, self.drm)

    def add_core(self, core):
        self.drm.cores.append(core)
        if self.generator is None:
            self.generator = self.sketch_method(self.tensor)

    def __next__(self):
        return next(self.generator)


def general_sketch(
    tensor: Tensor,
    left_drm: DRM,
    right_drm: DRM,
    orthogonalize: bool = False,
) -> SketchContainer:
    n_dims = len(tensor.shape)

    left_contractions = list(get_sketch_method(tensor, left_drm)(tensor))
    right_contractions = list(get_sketch_method(tensor, right_drm)(tensor))

    Psi_cores: ArrayList = []
    Omega_mats: ArrayList = []

    # Compute Omega matrices
    omega_method = OMEGA_METHODS[type(tensor)]
    for mu in range(n_dims - 1):
        omega_shape = (left_drm.rank[mu], right_drm.rank[::-1][mu])
        Omega_mats.append(
            omega_method(
                left_contractions[mu],
                right_contractions[mu],
                tensor=tensor,
                mu=mu,
                omega_shape=omega_shape,
            )
        )

    if orthogonalize:
        left_psi_drm = OrthogTTDRM(left_drm.rank, tensor)

    # Compute Psi cores
    psi_method = PSI_METHODS[type(tensor)]
    for mu in range(n_dims):
        if mu > 0:
            if orthogonalize:
                left_psi_drm.add_core(Psi_cores[-1])
                left_sketch = next(left_psi_drm)
            else:
                left_sketch = left_contractions[mu - 1]
            r1 = left_drm.rank[mu - 1]
        else:
            left_sketch = None
            r1 = 1
        if mu < n_dims - 1:
            right_sketch = right_contractions[mu]
            r2 = right_drm.rank[::-1][mu]
        else:
            right_sketch = None
            r2 = 1

        psi_shape = (r1, tensor.shape[mu], r2)
        Psi = psi_method(
            left_sketch, right_sketch, tensor=tensor, mu=mu, psi_shape=psi_shape
        )
        if orthogonalize and mu < n_dims - 1:
            Psi = orth_step(Psi, Omega_mats[mu])
        Psi_cores.append(Psi)
    return SketchContainer(Psi_cores, Omega_mats)


# def sum_sketch(
#     tensor: TensorSum,
#     left_drm: DRM,
#     right_drm: DRM,
#     orthogonalize: bool = False,
# ) -> SketchContainer:
#     """Sketch a tensor sum"""
#     left_rank = left_drm.rank
#     right_rank = right_drm.rank[::-1]
#     shape = left_drm.shape
#     sketch = SketchContainer.zero(shape, left_rank, right_rank)

#     for summand in tensor.tensors:
#         sketch_method = SKETCHING_METHODS_TENSOR[type(summand)]
#         sketch_summand = sketch_method(summand, left_drm, right_drm)  # type: ignore
#         sketch += sketch_summand

#     return sketch


SKETCHING_METHODS_TENSOR = {
    SparseTensor: sparse_sketch,
    TensorTrain: tensor_train_sketch,
    DenseTensor: dense_sketch,
    CPTensor: cp_sketch,
    TensorSum: sum_sketch,
}
