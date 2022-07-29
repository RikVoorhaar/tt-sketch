"""Interface for the streaming and orthogonal sketching algorithms"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import numpy.typing as npt

from tt_sketch.drm import (
    ALL_DRM,
    DenseGaussianDRM,
    SparseGaussianDRM,
    TensorTrainDRM,
)
from tt_sketch.drm_base import DRM, CanIncreaseRank, CanSlice
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.sketch_dispatch import SketchMethod, general_sketch
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchCP,
    CansketchDense,
    CansketchSparse,
    CansketchTT,
)
from tt_sketch.tensor import Tensor, TensorTrain
from tt_sketch.utils import (
    ArrayList,
    TTRank,
    left_mul_pinv,
    process_tt_rank,
    right_mul_pinv,
)

DEFAULT_DRM = {
    CansketchDense: DenseGaussianDRM,
    CansketchSparse: SparseGaussianDRM,
    CansketchTT: TensorTrainDRM,
    CansketchCP: TensorTrainDRM,
}

BlockedSketch = Dict[Tuple[int, int], SketchContainer]


def hmt_sketch(
    tensor: Tensor,
    rank: TTRank,
    seed: Optional[int] = None,
    drm_type: Optional[Type[DRM]] = None,
    drm: Optional[DRM] = None,
    return_drm: bool = False,
) -> TensorTrain:
    """
    Perform an orthogonal sketch of a tensor
    """
    d = len(tensor.shape)

    if seed is None:
        seed = np.mod(hash(np.random.uniform()), 2**32)

    if drm is None:
        if drm_type is None:
            drm_type = TensorTrainDRM
        rank = process_tt_rank(rank, tensor.shape, trim=True)
        drm = drm_type(rank, transpose=True, shape=tensor.shape, seed=seed)
    else:
        if tuple(drm.rank[::-1]) != rank:
            raise ValueError(
                f"Right rank {rank} does not match the rank of the DRM "
                f"{drm.rank}."
            )

    sketch = general_sketch(tensor, None, drm, method=SketchMethod.hmt)

    sketched = TensorTrain(sketch.Psi_cores)
    if return_drm:  # this really is mostly for testing purposes
        return sketched, drm, right_drm  # type: ignore
    else:
        return sketched


def orthogonal_sketch(
    tensor: Tensor,
    left_rank: TTRank,
    right_rank: TTRank,
    seed: Optional[int] = None,
    left_drm_type: Optional[Type[DRM]] = None,
    right_drm_type: Optional[Type[DRM]] = None,
    left_drm: Optional[DRM] = None,
    right_drm: Optional[DRM] = None,
    return_drm: bool = False,
) -> TensorTrain:
    """
    Perform an orthogonal sketch of a tensor
    """
    d = len(tensor.shape)

    right_rank_bigger = bool(np.all(np.array(left_rank) < np.array(right_rank)))
    if not right_rank_bigger:
        raise ValueError(
            f"The right rank needs to be larger than the left rank. "
            f"Left rank: {left_rank}, "
            f"right rank: {right_rank}"
        )

    if seed is None:
        seed = np.mod(hash(np.random.uniform()), 2**32)

    if left_drm is None:
        if left_drm_type is None:
            if right_drm_type is not None:
                left_drm_type = right_drm_type
            else:
                left_drm_type = TensorTrainDRM
        left_rank = process_tt_rank(left_rank, tensor.shape, trim=True)
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
                right_drm_type = TensorTrainDRM
        right_rank = process_tt_rank(right_rank, tensor.shape, trim=False)
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

    sketch = general_sketch(
        tensor, left_drm, right_drm, method=SketchMethod.orthogonal
    )

    sketched = TensorTrain(sketch.Psi_cores)
    if return_drm:  # this really is mostly for testing purposes
        return sketched, left_drm, right_drm  # type: ignore
    else:
        return sketched


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
    d = len(tensor.shape)

    left_rank_bigger = bool(np.all(np.array(left_rank) > np.array(right_rank)))
    right_rank_bigger = bool(np.all(np.array(left_rank) < np.array(right_rank)))
    if not left_rank_bigger and not right_rank_bigger:
        raise ValueError(
            f"Left ranks or right ranks must be conistently larger or smaller "
            f"than the other. Left rank: {left_rank}, "
            f"right rank: {right_rank}"
        )

    if seed is None:
        seed = np.mod(hash(np.random.uniform()), 2**32)

    if left_drm is None:
        if left_drm_type is None:
            if right_drm_type is not None:
                left_drm_type = right_drm_type
            else:
                left_drm_type = TensorTrainDRM
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
                right_drm_type = TensorTrainDRM
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

    sketch = general_sketch(
        tensor, left_drm, right_drm, method=SketchMethod.streaming
    )

    sketched = SketchedTensorTrain(sketch, left_drm, right_drm)
    if return_drm:  # this really is mostly for testing purposes
        return sketched, left_drm, right_drm  # type: ignore
    else:
        return sketched


@dataclass
class SketchedTensorTrain(Tensor):
    """
    Container for storing the output of the streaming sketch

    Stores the result of the sketch as well as the DRMs used for the sketching.
    Can be cheaply converted to a tensor train, or the sketch can be efficiently
    updated using the ``__add__`` method.
    """

    sketch_: SketchContainer
    left_drm: DRM
    right_drm: DRM

    @property
    def left_rank(self) -> Tuple[int, ...]:
        return self.left_drm.rank

    @property
    def right_rank(self) -> Tuple[int, ...]:
        return self.right_drm.rank[::-1]

    @property
    def Psi_cores(self) -> ArrayList:
        return self.sketch_.Psi_cores

    @property
    def size(self) -> int:
        total_Psi_size = sum(Psi.size for Psi in self.Psi_cores)
        total_Omega_size = sum(Omega.size for Omega in self.Omega_mats)
        return total_Psi_size + total_Omega_size

    @property
    def Omega_mats(self) -> ArrayList:
        return self.sketch_.Omega_mats

    def __post_init__(self):
        self.shape = self.sketch_.shape

    def C_cores(self, direction="auto") -> ArrayList:
        return assemble_sketched_tt(self.sketch_, direction=direction)

    @property
    def T(self) -> SketchedTensorTrain:
        new_sketch = self.sketch_.T
        return self.__class__(new_sketch, self.right_drm.T, self.left_drm.T)

    def to_tt(self) -> TensorTrain:
        return TensorTrain(self.C_cores())

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self.to_tt().to_numpy()

    def __repr__(self) -> str:
        return (
            f"<Sketched tensor train of shape {self.shape} with left-rank "
            f"{self.left_rank} and right-rank {self.right_rank} "
            f"at {hex(id(self))}>"
        )

    def __add__(self, other: Tensor) -> SketchedTensorTrain:
        other_sketch = stream_sketch(
            other,
            self.left_rank,
            self.right_rank,
            left_drm=self.left_drm,
            right_drm=self.right_drm,
        )
        new_sketch = self.sketch_ + other_sketch.sketch_
        return self.__class__(new_sketch, self.left_drm, self.right_drm)

    def increase_rank(
        self,
        tensor: Tensor,
        new_left_rank: TTRank,
        new_right_rank: TTRank,
    ) -> SketchedTensorTrain:
        """Increase the rank of the approximation by performing a new sketch.

        Requires DRM with support for the ``CanIncreaseRank`` protocol, which
        currently is only supported by ``SparseGaussianDRM``.
        """
        new_left_rank = process_tt_rank(new_left_rank, tensor.shape, trim=False)
        new_right_rank = process_tt_rank(
            new_right_rank, tensor.shape, trim=False
        )
        for drm in (self.left_drm, self.right_drm):
            if not isinstance(drm, CanSlice):
                drm_name = drm.__class__.__name__
                raise ValueError(
                    f"Increasing rank is not supported for DRM {drm_name}"
                )

        n_dims = len(tensor.shape)
        left_rank_slices = [
            (0,) * (n_dims - 1),
            self.left_drm.rank,
            new_left_rank,
        ]
        right_rank_slices = [
            (0,) * (n_dims - 1),
            self.right_drm.rank[::-1],
            new_right_rank,
        ]
        left_drm = self.left_drm.increase_rank(new_left_rank)  # type: ignore
        right_drm = self.right_drm.increase_rank(new_right_rank)  # type: ignore

        sketch_dict = _blocked_stream_sketch_components(
            tensor,
            left_drm,
            right_drm,
            left_rank_slices,
            right_rank_slices,
            excluded_entries=[(0, 0)],
        )

        sketch_dict[(0, 0)] = self.sketch_
        sketch = _assemble_blocked_stream_sketches(
            left_rank_slices, right_rank_slices, tensor.shape, sketch_dict
        )

        return self.__class__(sketch, left_drm, right_drm)

    def __mul__(self, other: float) -> SketchedTensorTrain:
        return self.__class__(
            self.sketch_ * other, self.left_drm, self.right_drm
        )

    def dot(self, other: Tensor, reverse=False) -> float:
        return self.to_tt().dot(other, reverse)


def _blocked_stream_sketch_components(
    tensor: Tensor,
    left_rm: CanSlice,
    right_drm: CanSlice,
    left_rank_slices: List[Tuple[int, ...]],
    right_rank_slices: List[Tuple[int, ...]],
    excluded_entries: Optional[Sequence[Tuple[int, int]]] = None,
) -> BlockedSketch:
    if excluded_entries is None:
        excluded_entries = []
    block_left_sketches = [
        left_rm.slice(rank1, rank2)
        for rank1, rank2 in zip(left_rank_slices[:-1], left_rank_slices[1:])
    ]
    block_right_sketches = [
        right_drm.slice(rank1, rank2)
        for rank1, rank2 in zip(right_rank_slices[:-1], right_rank_slices[1:])
    ]

    # Compute all the sketches
    sketch_dict = {}
    for i, left_sketch_slice in enumerate(block_left_sketches):
        for j, right_sketch_slice in enumerate(block_right_sketches):
            if (i, j) in excluded_entries:
                continue
            sketch_block = general_sketch(
                tensor,
                left_sketch_slice,
                right_sketch_slice,
                method=SketchMethod.streaming,
            )
            sketch_dict[(i, j)] = sketch_block

    return sketch_dict


def assemble_sketched_tt(
    sketch: SketchContainer,
    direction="auto",
) -> ArrayList:
    """Reconstructs a TT from a sketch, using Psi and Omega matrices."""
    tt_cores = []
    if direction == "auto":
        left_rank_bigger = np.all(
            np.array(sketch.left_rank) > np.array(sketch.right_rank)
        )
        direction = "left" if left_rank_bigger else "right"

    if direction == "right":
        for Psi, Omega in zip(sketch.Psi_cores[:-1], sketch.Omega_mats):
            Psi_shape = Psi.shape
            Psi_mat = Psi.reshape(Psi_shape[0] * Psi_shape[1], Psi_shape[2])
            try:
                Psi_Omega_pinv = right_mul_pinv(Psi_mat, Omega)
            except ValueError:
                print(Psi.shape, Omega.shape)
                raise
            core = Psi_Omega_pinv.reshape(
                Psi_shape[0], Psi_shape[1], Omega.shape[0]
            )
            tt_cores.append(core)
        tt_cores.append(sketch.Psi_cores[-1])
    elif direction == "left":
        tt_cores.append(sketch.Psi_cores[0])
        for Psi, Omega in zip(sketch.Psi_cores[1:], sketch.Omega_mats):
            Psi_shape = Psi.shape
            Psi_mat = Psi.reshape(Psi_shape[0], Psi_shape[1] * Psi_shape[2])
            try:
                Omega_pinv_Psi = left_mul_pinv(Omega, Psi_mat)
            except ValueError:
                print(Psi.shape, Omega.shape)
                raise
            core = Omega_pinv_Psi.reshape(
                Omega.shape[1], Psi_shape[1], Psi_shape[2]
            )
            tt_cores.append(core)
    else:
        raise ValueError(f"Unknown direction {direction}")

    return tt_cores


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

    It's use is mainly theoretical, since this this would only be faster in a
    distributed setting (which isn't properly supported).
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
