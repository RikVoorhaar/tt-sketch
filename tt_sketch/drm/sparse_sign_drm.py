from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from tt_sketch.drm.fast_lazy_gaussian import (
    inds_to_sparse_sign,
)  # type: ignore
from tt_sketch.drm_base import handle_transpose, CanSlice
from tt_sketch.sketching_methods.sparse_sketch import CansketchSparse
from tt_sketch.tensor import SparseTensor
from tt_sketch.utils import ArrayList


class SparseSignDRM(CansketchSparse, CanSlice):
    def __init__(
        self,
        rank: Union[Tuple[int, ...], int],
        shape: Tuple[int, ...],
        transpose: bool,
        seed: Optional[int] = None,
        num_non_zero_per_row: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> None:
        super().__init__(rank, shape, transpose, seed=seed, **kwargs)
        if num_non_zero_per_row is None:
            num_non_zero_per_row = self.true_rank
        self.nnz = num_non_zero_per_row

    @handle_transpose
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayList:
        d = len(tensor.shape)
        sketching_mats = []
        for mu in range(d - 1):
            shape = tensor.shape[: mu + 1]
            sketch_seed = np.mod(
                mu + self.seed, 2 ** 63, dtype=np.uint64
            )  # ensure safe casting to uint
            sketch_mat = inds_to_sparse_sign(
                tensor.indices[: mu + 1],
                shape,
                self.true_rank[mu],
                self.rank_min[mu],
                self.rank_max[mu],
                self.nnz[mu],
                sketch_seed,
            )
            sketching_mats.append(sketch_mat.T)
        return sketching_mats
