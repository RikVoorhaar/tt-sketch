from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import SeedSequence
from tt_sketch.drm_base import CanIncreaseRank, handle_transpose
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchDense,
    CansketchSparse,
    CansketchTT,
)
from tt_sketch.tensor import DenseTensor, SparseTensor, TensorTrain
from tt_sketch.utils import ArrayGenerator, ArrayList, random_normal


class DenseGaussianDRM(
    CansketchTT, CansketchSparse, CansketchDense, CanIncreaseRank
):
    """Dense Gaussian DRM.

    The DRM is stored as a list of matrices, one for each mode."""

    sketching_mats: ArrayList

    def __init__(
        self,
        rank: Union[Tuple[int, ...], int],
        shape: Tuple[int, ...],
        transpose: bool,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(rank, shape, transpose, seed=seed, **kwargs)

        self.sketching_mats = []
        dim_prod = 1
        shape_sketch = self.shape
        if transpose:
            shape_sketch = shape_sketch[::-1]

        # seq = SeedSequence(self.seed)
        # seeds = seq.generate_state(len(self.true_rank))
        for mu, (r, n) in enumerate(
            zip(self.true_rank, shape_sketch[:-1])
        ):
            dim_prod *= n

            # TODO: make this correctly work with rank_increase and random_normal
            np.random.seed(seed)
            seed_offset = hash(np.random.uniform(0, dim_prod))
            np.random.seed(np.mod(self.seed + seed_offset, 2**32 - 1))

            # sketching_mat = random_normal(shape=(r, dim_prod), seed=seed)
            sketching_mat = np.random.normal(size=(r, dim_prod))
            sketching_mat = sketching_mat[self.rank_min[mu] : self.rank_max[mu]]
            self.sketching_mats.append(sketching_mat)

    @handle_transpose
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayGenerator:
        d = len(tensor.shape)
        for mu in range(d - 1):
            shape = tensor.shape[: mu + 1]
            inds = tensor.indices[: mu + 1]
            inds = np.ravel_multi_index(inds, shape)  # type: ignore
            sketching_vec = self.sketching_mats[mu][:, inds]
            yield sketching_vec

    @handle_transpose
    def sketch_tt(self, tensor: TensorTrain) -> ArrayGenerator:
        r"""Contract sketching matrix with :math:`Phi_{\leq \mu}`"""
        partial_contracts = tensor.partial_dense("lr")
        for sm, pc in zip(self.sketching_mats, partial_contracts):
            sketch = (sm @ pc).T
            yield sketch

    @handle_transpose
    def sketch_dense(self, tensor: DenseTensor) -> ArrayGenerator:
        for mat in self.sketching_mats:
            yield mat
