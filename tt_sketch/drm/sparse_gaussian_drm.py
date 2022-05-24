from typing import Optional, Tuple, Union

import numpy as np
from tt_sketch.drm.fast_lazy_gaussian import inds_to_normal  # type: ignore
from tt_sketch.drm_base import CanIncreaseRank, handle_transpose
from tt_sketch.sketching_methods.abstract_methods import CansketchSparse
from tt_sketch.tensor import SparseTensor
from tt_sketch.utils import ArrayGenerator


class SparseGaussianDRM(CansketchSparse, CanIncreaseRank):
    """'Sparse' Gaussian DRM

    Mathematically equivalent ``DenseGaussianDRM``, but entries of the DRM
    are computed lazily/on-demand using a hashing algorithm. This makes
    it computationally feasible for very sparse tensors.
    """

    def __init__(
        self,
        rank: Union[Tuple[int, ...], int],
        shape: Tuple[int, ...],
        transpose: bool,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(rank, shape, transpose, seed=seed, **kwargs)

    @handle_transpose
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayGenerator:
        d = len(tensor.shape)
        for mu in range(d - 1):
            shape = tensor.shape[: mu + 1]
            sketch_seed = np.mod(
                mu + self.seed, 2**63, dtype=np.uint64
            )  # ensure safe casting to uint
            sketch_mat = inds_to_normal(
                tensor.indices[: mu + 1],
                shape,
                self.rank_min[mu],
                self.rank_max[mu],
                sketch_seed,
            )
            yield sketch_mat.T
