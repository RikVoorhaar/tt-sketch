from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Union

from numpy import mod
from numpy.random import uniform

from tt_sketch.utils import TTRank, process_tt_rank


class DRM(ABC):
    """Implements an abstract dimension reduction matrix"""

    rank: Tuple[int, ...]  # The rank of the DRM. (Size of slice for blocked)
    rank_min: Tuple[int, ...]  # For blocked sketch, the start of rank slice
    rank_max: Tuple[int, ...]  # For blocked sketch, the end of rank slice
    true_rank: Tuple[int, ...]  # For blocked sketch, the true size before slice
    shape: Tuple[int, ...]  # shape of tensor
    transpose: bool  # If false, left sketch, if true right sketch
    seed: int

    def __init__(
        self,
        rank: TTRank,
        shape: Tuple[int, ...],
        transpose: bool,
        seed: Optional[int] = None,
        rank_min: Optional[Tuple[int, ...]] = None,
        rank_max: Optional[Tuple[int, ...]] = None,
        true_rank: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> None:
        self.transpose = transpose
        rank = process_tt_rank(rank, shape, trim=False)
        if true_rank is None:
            self.true_rank = rank
        else:
            self.true_rank = true_rank

        if rank_min is None:
            self.rank_min = (0,) * (len(shape) - 1)
        else:
            self.rank_min = rank_min
        if rank_max is None:
            self.rank_max = rank
        else:
            self.rank_max = rank_max

        if transpose:
            self.true_rank = self.true_rank[::-1]
            self.rank_min = self.rank_min[::-1]
            self.rank_max = self.rank_max[::-1]
        self.rank = tuple(
            r2 - r1 for r1, r2 in zip(self.rank_min, self.rank_max)
        )

        self.shape = shape
        if seed is None:
            seed = hash(uniform())
        seed = mod(seed, 2**32 - 1)
        self.seed = seed  # type: ignore

    @property
    def T(self):
        transposed = deepcopy(self)
        transposed.transpose = not self.transpose
        transposed.true_rank = transposed.true_rank[::-1]
        transposed.rank_min = transposed.rank_min[::-1]
        transposed.rank_max = transposed.rank_max[::-1]
        transposed.rank = transposed.rank[::-1]
        return transposed

    def __repr__(self) -> str:
        if self.transpose:
            direction = "Right"
        else:
            direction = "Left"
        return (
            f"<{direction} {self.__class__.__name__} of rank {self.rank}"
            f" and shape {self.shape} at {hex(id(self))}>"
        )


class CanSlice(DRM):
    """Mixin telling that the DRM supports slicing.

    Default implementation is inefficient, and in many cases should be
    overloaded."""

    def slice(
        self, start_rank: Tuple[int, ...], end_rank: Tuple[int, ...]
    ) -> DRM:
        if self.transpose:
            # If transpose, the rank will be reversed during init.
            # To keep rank the same, need to reverse it here as well.
            new_true_rank = self.true_rank[::-1]
        else:
            new_true_rank = self.true_rank
        return self.__class__(
            rank=self.rank,
            shape=self.shape,
            transpose=self.transpose,
            seed=self.seed,
            rank_min=start_rank,
            rank_max=end_rank,
            true_rank=new_true_rank,
        )


class CanIncreaseRank(CanSlice):
    """Mixin telling that the DRM supports rank increase and slicing.

    Standard implementations of increase_rank and slice are not optimal, and
    should be overloaded for good performance."""

    def increase_rank(self, new_rank: Tuple[int, ...]) -> DRM:
        return self.__class__(new_rank, self.shape, self.transpose, self.seed)


def handle_transpose(sketch: Callable) -> Callable:
    """Decorator to handle transpose of sketch.

    This way we only have to implement sketches on the left, and right-sketches
    are handled automatically."""

    def wrapper(self, tensor):
        if self.shape != tensor.shape:
            # Catch this shape mismatch early, because it can cause unexpected
            # behavior otherwise
            raise ValueError(
                f"""Shape {self.shape} of DRM doesn't match tensor's shape
                {tensor.shape}"""
            )
        if self.transpose:
            tensor = tensor.T

        sketching_mats = sketch(self, tensor)
        if self.transpose:
            sketching_mats = list(sketching_mats)[::-1]
        for mat in sketching_mats:
            yield mat

    return wrapper
