from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from tt_sketch.sketching_methods.sparse_sketch import CansketchSparse
from tt_sketch.sketching_methods.tensor_train_sketch import CansketchTT
from tt_sketch.sketching_methods.cp_sketch import CansketchCP
from tt_sketch.drm_base import handle_transpose, CanSlice
from tt_sketch.tensor import SparseTensor, TensorTrain, CPTensor
from tt_sketch.utils import ArrayList


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
        tt = TensorTrain.random(tt_shape, tt_rank, self.seed)
        self.cores = tt.cores[:-1]

    @handle_transpose
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayList:
        core_slices = [
            core_slice[:, tensor.indices[i], :]
            for i, core_slice in enumerate(self.cores)
        ]
        sketching_mats = []
        for mu, core_slice in enumerate(core_slices):
            if mu == 0:
                lr_contract = core_slice.reshape(core_slice.shape[1:])
            else:
                lr_contract = np.einsum("ijk,ji->jk", core_slice, lr_contract)
            sketching_mats.append(
                lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]
            )

        sketching_mats = [s.T for s in sketching_mats]
        return sketching_mats

    @handle_transpose
    def sketch_tt(self, tensor: TensorTrain) -> ArrayList:
        n_dims = len(self.shape)

        sketching_mats = []
        for mu in range(n_dims - 1):
            tensor_core = tensor.cores[mu]
            drm_core = self.cores[mu]
            if mu == 0:
                lr_contract = np.einsum("ijk,ijl->kl", tensor_core, drm_core)
            else:
                lr_contract = np.einsum(
                    "ij,ikl,jkm->lm", lr_contract, tensor_core, drm_core
                )
            sketching_mats.append(
                lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]
            )

        return sketching_mats

    @handle_transpose
    def sketch_cp(self, tensor: CPTensor) -> ArrayList:
        n_dims = len(self.shape)

        sketching_mats = []
        for mu in range(n_dims - 1):
            tensor_core = tensor.cores[mu]
            drm_core = self.cores[mu]
            if mu == 0:
                lr_contract = np.einsum("ij,lik->jk", tensor_core, drm_core)
            else:
                lr_contract = np.einsum(
                    "ij,ki,jkl->il", lr_contract, tensor_core, drm_core
                )
            sketching_mats.append(
                lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]
            )

        return sketching_mats
