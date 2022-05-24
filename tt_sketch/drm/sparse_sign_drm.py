from typing import Optional, Tuple, Union

import numpy as np
from tt_sketch.drm.fast_lazy_gaussian import inds_to_sparse_sign  # type: ignore
from tt_sketch.drm_base import CanSlice, handle_transpose
from tt_sketch.sketching_methods.abstract_methods import CansketchSparse
from tt_sketch.tensor import SparseTensor
from tt_sketch.utils import ArrayGenerator


class SparseSignDRM(CansketchSparse, CanSlice):
    """
    Sparse DRM where each row is a vector with fixed number of +/-1 entries.

    The number of nonzero entries are determined by ``num_non_zero_per_row``.
    Like ``SparseGaussianDRM``, entries are computed lazily/on-demand using a
    hashing algorithm.
    """

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
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayGenerator:
        d = len(tensor.shape)
        for mu in range(d - 1):
            shape = tensor.shape[: mu + 1]
            sketch_seed = np.mod(
                mu + self.seed, 2**63, dtype=np.uint64
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
            yield sketch_mat.T
