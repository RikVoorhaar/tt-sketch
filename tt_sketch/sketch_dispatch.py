"""
Implements methods for dispatching sketching methods for tensors and DRMs.
"""

import enum
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.linalg

from tt_sketch.drm import TensorTrainDRM
from tt_sketch.drm_base import DRM
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchCP,
    CansketchDense,
    CansketchSparse,
    CansketchTT,
    CanSketchTucker,
)
from tt_sketch.sketching_methods.cp_sketch import sketch_omega_cp, sketch_psi_cp
from tt_sketch.sketching_methods.dense_sketch import (
    sketch_omega_dense,
    sketch_psi_dense,
)
from tt_sketch.sketching_methods.sparse_sketch import (
    sketch_omega_sparse,
    sketch_psi_sparse,
)
from tt_sketch.sketching_methods.tensor_train_sketch import (
    sketch_omega_tt,
    sketch_psi_tt,
)
from tt_sketch.sketching_methods.tucker_sketch import (
    sketch_omega_tucker,
    sketch_psi_tucker,
)
from tt_sketch.tensor import (
    CPTensor,
    DenseTensor,
    SparseTensor,
    Tensor,
    TensorSum,
    TensorTrain,
    TuckerTensor,
)
from tt_sketch.utils import ArrayList, right_mul_pinv

ABSTRACT_TENSOR_SKETCH_DISPATCH = {
    SparseTensor: CansketchSparse,
    TensorTrain: CansketchTT,
    DenseTensor: CansketchDense,
    CPTensor: CansketchCP,
    TuckerTensor: CanSketchTucker,
}

DRM_SKETCH_METHOD_DISPATCH = {
    SparseTensor: "sketch_sparse",
    TensorTrain: "sketch_tt",
    DenseTensor: "sketch_dense",
    CPTensor: "sketch_cp",
    TuckerTensor: "sketch_tucker",
}


OMEGA_METHODS = {
    SparseTensor: sketch_omega_sparse,
    TensorTrain: sketch_omega_tt,
    DenseTensor: sketch_omega_dense,
    CPTensor: sketch_omega_cp,
    TuckerTensor: sketch_omega_tucker,
}

PSI_METHODS = {
    SparseTensor: sketch_psi_sparse,
    TensorTrain: sketch_psi_tt,
    DenseTensor: sketch_psi_dense,
    CPTensor: sketch_psi_cp,
    TuckerTensor: sketch_psi_tucker,
}


def sketch_omega_sum(
    left_sketch_array: ArrayList,
    right_sketch_array: ArrayList,
    *,
    tensor: TensorSum,
    omega_shape: Tuple[int, int],
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
        )  # type: ignore
    return omega


OMEGA_METHODS[TensorSum] = sketch_omega_sum


def sketch_psi_sum(
    left_sketch_array: ArrayList,
    right_sketch_array: ArrayList,
    *,
    tensor: TensorSum,
    psi_shape: Tuple[int, int],
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
        )  # type: ignore
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
    Psi: npt.NDArray[np.float64], Omega: Optional[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Perform the orthogonalization step in the orthogonal sketching algorithm.
    """
    Psi_shape = Psi.shape
    final_right_rank = Psi_shape[2] if Omega is None else Omega.shape[0]
    Psi_mat = Psi.reshape((Psi_shape[0] * Psi_shape[1], Psi_shape[2]))
    if Omega is not None:
        Psi_mat = right_mul_pinv(Psi_mat, Omega)
    # Psi_mat, _ = np.linalg.qr(Psi_mat)
    Psi_mat, _ = scipy.linalg.qr(Psi_mat, mode="economic")
    Psi = Psi_mat.reshape(Psi_shape[0], Psi_shape[1], final_right_rank)
    return Psi


class OrthogTTDRM:
    """Represents the orthogonalized TT used as left-sketch for psi"""

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


class SketchMethod(enum.Enum):
    streaming = "streaming"
    orthogonal = "orthogonal"
    hmt = "hmt"


def general_sketch(
    tensor: Tensor,
    left_drm: Optional[DRM],
    right_drm: DRM,
    method: SketchMethod,
) -> SketchContainer:
    """General algorithm for sketching a tensor.

    Does the heavy lifting for both the streaming and orthogonal sketching
    algorithms."""
    n_dims = len(tensor.shape)

    if method != SketchMethod.hmt:
        if left_drm is None:
            raise ValueError(f"left_drm must be provided for method '{method}'")
        left_contractions = list(get_sketch_method(tensor, left_drm)(tensor))
    right_contractions = list(get_sketch_method(tensor, right_drm)(tensor))

    if left_drm is None:
        # This is just required for shape information in the case of HMT
        left_drm = right_drm.T

    Psi_cores: ArrayList = []

    # Compute Omega matrices
    Omega_mats: ArrayList = []
    if method != SketchMethod.hmt:
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
                )  # type: ignore
            )

    if method in (SketchMethod.hmt, SketchMethod.orthogonal):
        left_psi_drm = OrthogTTDRM(left_drm.rank, tensor)

    # Compute Psi cores
    psi_method = PSI_METHODS[type(tensor)]
    for mu in range(n_dims):
        if mu > 0:
            if method in (SketchMethod.hmt, SketchMethod.orthogonal):
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
        )  # type: ignore
        if mu < n_dims - 1:
            if method == SketchMethod.orthogonal:
                Psi = orth_step(Psi, Omega_mats[mu])
            elif method == SketchMethod.hmt:
                Psi = orth_step(Psi, None)
        Psi_cores.append(Psi)
    return SketchContainer(Psi_cores, Omega_mats)
