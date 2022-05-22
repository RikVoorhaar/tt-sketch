"""Interface for the streaming and orthogonal sketching algorithms"""
from __future__ import annotations

from functools import cached_property
from re import L
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import numpy.typing as npt

from tt_sketch.drm import (
    DenseGaussianDRM,
    SparseGaussianDRM,
    TensorTrainDRM,
    ALL_DRM,
)
from tt_sketch.drm_base import DRM, CanSlice, CanIncreaseRank
from tt_sketch.sketching_methods import (
    SKETCHING_METHODS,
    TENSOR_SKETCH_DISPATCH,
    CansketchDense,
    CansketchSparse,
    CansketchTT,
    CansketchCP,
    sum_sketch,
)
from tt_sketch.tensor import Tensor, TensorTrain, TensorSum, SketchedTensorTrain
from tt_sketch.utils import ArrayList, TTRank, process_tt_rank, right_mul_pinv
from tt_sketch.sketch_container import SketchContainer

DEFAULT_DRM = {
    CansketchDense: DenseGaussianDRM,
    CansketchSparse: SparseGaussianDRM,
    CansketchTT: TensorTrainDRM,
    CansketchCP: TensorTrainDRM,
}

BlockedSketch = Dict[Tuple[int, int], SketchContainer]


SKETCH_METHOD_SIGNATURE = Callable[[Tensor, DRM, DRM], SketchContainer]


def stream_sketch(
    tensor: Tensor,
    left_rank: TTRank,
    right_rank: TTRank,
    seed: Optional[int] = None,
    left_drm_type: Optional[Type[DRM]] = None,
    right_drm_type: Optional[Type[DRM]] = None,
    left_drm: Optional[DRM] = None,
    right_drm: Optional[DRM] = None,
    return_drm: bool = False,
) -> SketchedTensorTrain:
    """
    Perform a streaming sketch of a tensor
    """
    sketch_method, default_drm_type = _get_sketch_method_default_drm(tensor)
    d = len(tensor.shape)

    left_rank_bigger = bool(np.all(np.array(left_rank) > np.array(right_rank)))
    right_rank_bigger = bool(np.all(np.array(left_rank) < np.array(right_rank)))
    if not left_rank_bigger and not right_rank_bigger:
        raise ValueError(
            f"Left ranks or right ranks must be conistently larger or smaller "
            f"than the other. Left rank: {left_rank}, "
            f"right side rank: {right_rank}"
        )

    if seed is None:
        seed = np.mod(hash(np.random.uniform()), 2**32)

    if left_drm is None:
        if left_drm_type is None:
            if right_drm_type is not None:
                left_drm_type = right_drm_type
            else:
                left_drm_type = default_drm_type
        left_rank = process_tt_rank(
            left_rank, tensor.shape, trim=right_rank_bigger
        )
        left_drm = left_drm_type(
            left_rank, transpose=False, shape=tensor.shape, seed=seed
        )
    else:
        if left_drm.rank != left_rank:
            raise ValueError(
                f"Left rank {left_rank} does not match the rank of the DRM "
                f"{left_drm.rank}."
            )

    if right_drm is None:
        if right_drm_type is None:
            if left_drm_type is not None:
                right_drm_type = left_drm_type
            else:
                right_drm_type = default_drm_type
        right_rank = process_tt_rank(
            right_rank, tensor.shape, trim=left_rank_bigger
        )
        right_seed = np.mod(seed + hash(str(d)), 2**32)
        right_drm = right_drm_type(
            right_rank, transpose=True, shape=tensor.shape, seed=right_seed
        )
    else:
        if tuple(right_drm.rank[::-1]) != right_rank:
            raise ValueError(
                f"Right rank {right_rank} does not match the rank of the DRM "
                f"{right_drm.rank}."
            )

    sketch = sketch_method(tensor, left_drm, right_drm)

    sketched = SketchedTensorTrain(sketch)
    if return_drm:  # this really is mostly for testing purposes
        return sketched, left_drm, right_drm  # type: ignore
    else:
        return sketched


def _get_sketch_method_default_drm(
    tensor: Tensor,
) -> Tuple[SKETCH_METHOD_SIGNATURE, Type[DRM]]:
    """Figure out which sketching method makes the most sense for a tensor"""
    if isinstance(tensor, TensorSum):
        sketch_method = sum_sketch
        matching_drms = set()
        for X in tensor.tensors:
            _, drm = _get_sketch_method_default_drm(X)
            matching_drms.add(drm)
        if len(matching_drms) > 0:
            return sketch_method, list(matching_drms)[0]  # type: ignore
    else:
        for tensor_type, sketch_type in TENSOR_SKETCH_DISPATCH.items():
            if isinstance(tensor, tensor_type):
                sketch_method = SKETCHING_METHODS[sketch_type]  # type: ignore
                default_drm_type = DEFAULT_DRM[sketch_type]
                return sketch_method, default_drm_type  # type: ignore

    # No matching method found
    raise ValueError(
        f"""No sketching methods available for tensor of type
        {type(tensor)}"""
    )


def _blocked_stream_sketch_components(
    tensor: Tensor,
    left_sketch: CanSlice,
    right_sketch: CanSlice,
    left_rank_slices: List[Tuple[int, ...]],
    right_rank_slices: List[Tuple[int, ...]],
    excluded_entries: Optional[Sequence[Tuple[int, int]]] = None,
) -> BlockedSketch:
    shape = tensor.shape
    sketch_method, _ = _get_sketch_method_default_drm(tensor)
    if excluded_entries is None:
        excluded_entries = []
    block_left_sketches = [
        left_sketch.slice(rank1, rank2)
        for rank1, rank2 in zip(left_rank_slices[:-1], left_rank_slices[1:])
    ]
    block_right_sketches = [
        right_sketch.slice(rank1, rank2)
        for rank1, rank2 in zip(right_rank_slices[:-1], right_rank_slices[1:])
    ]

    # Compute all the sketches
    sketch_dict = {}
    for i, left_sketch_slice in enumerate(block_left_sketches):
        for j, right_sketch_slice in enumerate(block_right_sketches):
            if (i, j) in excluded_entries:
                continue
            sketch_block = sketch_method(
                tensor, left_sketch_slice, right_sketch_slice
            )
            sketch_dict[(i, j)] = sketch_block

    return sketch_dict


def _assemble_blocked_stream_sketches(
    left_rank_slices: List[Tuple[int, ...]],
    right_rank_slices: List[Tuple[int, ...]],
    shape: Tuple[int, ...],
    sketch_dict: BlockedSketch,
) -> SketchContainer:
    left_rank = tuple(left_rank_slices[-1])
    right_rank = tuple(right_rank_slices[-1])

    sketch = SketchContainer.zero(shape, left_rank, right_rank)
    for (i, j), sketch_block in sketch_dict.items():
        left_rank1 = (0,) + left_rank_slices[i]
        left_rank2 = (1,) + left_rank_slices[i + 1]
        right_rank1 = right_rank_slices[j] + (0,)
        right_rank2 = right_rank_slices[j + 1] + (1,)
        for mu, Psi in enumerate(sketch_block.Psi_cores):
            sketch.Psi_cores[mu][
                left_rank1[mu] : left_rank2[mu],
                :,
                right_rank1[mu] : right_rank2[mu],
            ] = Psi
        for mu, Omega in enumerate(sketch_block.Omega_mats):
            sketch.Omega_mats[mu][
                left_rank1[mu + 1] : left_rank2[mu + 1],
                right_rank1[mu] : right_rank2[mu],
            ] = Omega

    return sketch


def get_drm_capabilities():
    """List what all the DRMs are capable of"""
    all_capabilities = {}
    for drm in ALL_DRM:
        drm_capabilities = {}
        for capability in (
            CanSlice,
            CanIncreaseRank,
            CansketchSparse,
            CansketchDense,
            CansketchTT,
        ):
            drm_capabilities[capability.__name__] = issubclass(drm, capability)
        all_capabilities[drm.__name__] = drm_capabilities
    return all_capabilities


def blocked_stream_sketch(
    tensor: Tensor,
    left_drm: CanSlice,
    right_drm: CanSlice,
    left_rank_slices: List[Tuple[int, ...]],
    right_rank_slices: List[Tuple[int, ...]],
) -> SketchContainer:
    """Do a blocked sketch.

    Parameters
    ----------
    tensor
    left_drm
    right_drm
    left_rank_slices
        Instructions how to slice up the left-sketch. If the t--rank of the
        left-sketch is for example (6,8,10), then we can set ``left_rank_slices
        = [(0,0,0), (3,4,5), (6,8,10)]`` to split the left-sketch into two
        equally sized blocks. The number of slices is unlimited.
    right_rank_slices
        Same as ``left_rank_slices`` but pertaining to the right-sketch.
    """
    for drm in (left_drm, right_drm):
        if not isinstance(drm, CanSlice):
            drm_name = drm.__class__.__name__
            raise ValueError(f"Blocked sketch not supported for DRM {drm_name}")

    sketch_dict = _blocked_stream_sketch_components(
        tensor,
        left_drm,
        right_drm,
        left_rank_slices,
        right_rank_slices,
    )

    sketch = _assemble_blocked_stream_sketches(
        left_rank_slices,
        right_rank_slices,
        tensor.shape,
        sketch_dict,
    )

    return sketch
