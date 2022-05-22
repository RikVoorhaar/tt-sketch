from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from tt_sketch.sketching_methods.abstract_methods import (
    CansketchSparse,
    CansketchTT,
    CansketchCP,
)
from tt_sketch.drm_base import handle_transpose, CanSlice
from tt_sketch.tensor import SparseTensor, TensorTrain, CPTensor
from tt_sketch.utils import ArrayList, ArrayGenerator


# TODO: Store DRM as a tensor.TensorTrain
class TensorTrainDRM(CansketchSparse, CansketchTT, CansketchCP, CanSlice):
    """Sketcher using Tensor Trains"""

    cores: ArrayList

    def __init__(
        self,
        rank: Union[Tuple[int, ...], int],
        shape: Tuple[int, ...],
        transpose: bool,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(rank, shape, transpose, seed=seed, **kwargs)
        if transpose:
            tt_shape = self.shape[::-1]
            tt_rank = self.true_rank
        else:
            tt_shape = self.shape
            tt_rank = self.true_rank
        if "cores" not in kwargs:
            tt = TensorTrain.random(tt_shape, tt_rank, self.seed)
            self.cores = tt.cores[:-1]
        else:
            self.cores = kwargs["cores"]

    @handle_transpose
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayGenerator:
        for mu, core in enumerate(self.cores):
            core_slice = core[:, tensor.indices[mu], :]
            if mu == 0:
                lr_contract = core_slice.reshape(core_slice.shape[1:])
            else:
                lr_contract = np.einsum("ijk,ji->jk", core_slice, lr_contract)
            sketch_mat = lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]
            yield sketch_mat.T

    @handle_transpose
    def sketch_tt(self, tensor: TensorTrain) -> ArrayGenerator:
        n_dims = len(self.shape)

        for mu in range(n_dims - 1):
            tensor_core = tensor.cores[mu]
            drm_core = self.cores[mu]
            if mu == 0:
                lr_contract = np.einsum("ijk,ijl->kl", tensor_core, drm_core)
            else:
                lr_contract = np.einsum(
                    "ij,ikl,jkm->lm", lr_contract, tensor_core, drm_core
                )
            yield lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]

    @handle_transpose
    def sketch_cp(self, tensor: CPTensor) -> ArrayGenerator:
        n_dims = len(self.shape)

        for mu in range(n_dims - 1):
            tensor_core = tensor.cores[mu]
            drm_core = self.cores[mu]
            if mu == 0:
                lr_contract = np.einsum("ij,lik->jk", tensor_core, drm_core)
            else:
                lr_contract = np.einsum(
                    "ij,ki,jkl->il", lr_contract, tensor_core, drm_core
                )
            yield lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]
