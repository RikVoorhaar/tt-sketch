from __future__ import annotations
from copy import deepcopy

from typing import Optional, Tuple

import numpy as np

from tt_sketch.utils import ArrayList


class SketchContainer:
    """
    Container class for the Psi_cores and Omega_mats often used internally.
    """

    shape: Tuple[int, ...]
    left_rank: Tuple[int, ...]
    right_rank: Tuple[int, ...]
    Psi_cores: ArrayList
    Omega_mats: ArrayList

    def __init__(
        self,
        Psi_cores: ArrayList,
        Omega_mats: ArrayList,
        shape: Optional[Tuple[int, ...]] = None,
        left_rank: Optional[Tuple[int, ...]] = None,
        right_rank: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.Psi_cores = Psi_cores
        self.Omega_mats = Omega_mats

        # Infer shapes and ranks from Psi_cores
        if shape is None:
            shape = tuple(Psi.shape[1] for Psi in Psi_cores)
        if left_rank is None:
            left_rank = tuple(Psi.shape[0] for Psi in Psi_cores[1:])
        if right_rank is None:
            right_rank = tuple(Psi.shape[2] for Psi in Psi_cores[:-1])
        self.shape = shape
        self.left_rank = left_rank
        self.right_rank = right_rank

    @classmethod
    def zero(
        cls,
        shape: Tuple[int, ...],
        left_rank: Tuple[int, ...],
        right_rank: Tuple[int, ...],
    ) -> SketchContainer:
        Psi_cores = []
        for r1, n, r2 in zip((1,) + left_rank, shape, right_rank + (1,)):
            Psi_cores.append(np.zeros((r1, n, r2)))

        Omega_mats = []
        for r1, r2 in zip(left_rank, right_rank):
            Omega_mats.append(np.zeros((r1, r2)))

        return cls(Psi_cores, Omega_mats, shape, left_rank, right_rank)

    def __add__(self, other: SketchContainer) -> SketchContainer:
        Psi_cores_new = [
            Psi1 + Psi2 for Psi1, Psi2 in zip(self.Psi_cores, other.Psi_cores)
        ]
        Omega_mats_new = [
            Omega1 + Omega2
            for Omega1, Omega2 in zip(self.Omega_mats, other.Omega_mats)
        ]
        return self.__class__(Psi_cores_new, Omega_mats_new)

    @property
    def T(self) -> SketchContainer:
        Psi_cores_new = [Psi.transpose(2, 1, 0) for Psi in self.Psi_cores[::-1]]
        Omega_mats_new = [Omega.T for Omega in self.Omega_mats[::-1]]
        return self.__class__(Psi_cores_new, Omega_mats_new)

    def __mul__(self, other: float) -> SketchContainer:
        new_Psi_cores = deepcopy(self.Psi_cores)
        new_Psi_cores[0] *= other
        return self.__class__(new_Psi_cores, self.Omega_mats)
