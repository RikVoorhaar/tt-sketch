from typing import Callable, Dict, Type
from tt_sketch.sketching_methods.sparse_sketch import sparse_sketch
from tt_sketch.sketching_methods.tensor_train_sketch import tensor_train_sketch
from tt_sketch.sketching_methods.cp_sketch import cp_sketch
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchCP,
    CansketchDense,
    CansketchSparse,
    CansketchTT,
)

from tt_sketch.sketching_methods.dense_sketch import dense_sketch
from tt_sketch.tensor import (
    DenseTensor,
    SparseTensor,
    TensorTrain,
    CPTensor,
    TensorSum,
)
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.drm_base import DRM

SKETCHING_METHODS = {
    CansketchSparse: sparse_sketch,
    CansketchTT: tensor_train_sketch,
    CansketchDense: dense_sketch,
    CansketchCP: cp_sketch,
}

TENSOR_SKETCH_DISPATCH = {
    SparseTensor: CansketchSparse,
    TensorTrain: CansketchTT,
    DenseTensor: CansketchDense,
    CPTensor: CansketchCP,
}


def sum_sketch(
    tensor: TensorSum,
    left_drm: DRM,
    right_drm: DRM,
) -> SketchContainer:
    """Sketch a tensor sum"""
    left_rank = left_drm.rank
    right_rank = right_drm.rank[::-1]
    shape = left_drm.shape
    sketch = SketchContainer.zero(shape, left_rank, right_rank)

    for summand in tensor.tensors:
        sketch_method = SKETCHING_METHODS_TENSOR[type(summand)]
        sketch_summand = sketch_method(summand, left_drm, right_drm)  # type: ignore
        sketch += sketch_summand

    return sketch


SKETCHING_METHODS_TENSOR = {
    SparseTensor: sparse_sketch,
    TensorTrain: tensor_train_sketch,
    DenseTensor: dense_sketch,
    CPTensor: cp_sketch,
    TensorSum: sum_sketch,
}
