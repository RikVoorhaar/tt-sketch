from typing import Optional, Tuple, Union

import numpy as np
from tt_sketch.drm_base import CanSlice, handle_transpose
from tt_sketch.sketching_methods.abstract_methods import (
    CansketchCP,
    CansketchDense,
    CansketchSparse,
    CansketchTT,
    CanSketchTucker,
)
from tt_sketch.tensor import (
    CPTensor,
    DenseTensor,
    SparseTensor,
    TensorTrain,
    TuckerTensor,
)
from tt_sketch.utils import ArrayGenerator, ArrayList


# TODO: Store DRM as a tensor.TensorTrain
class TensorTrainDRM(
    CansketchSparse,
    CansketchTT,
    CansketchCP,
    CanSlice,
    CansketchDense,
    CanSketchTucker,
):
    """
    Tensor train DRM. Sketches with partial contractions of a fixed TT.
    """

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
            tt = TensorTrain.random(
                tt_shape, tt_rank, self.seed, norm_goal="norm-preserve"
            )
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
                    "ij,ikl,jkm->lm",
                    lr_contract,
                    tensor_core,
                    drm_core,
                    optimize="optimal",
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
                    "ij,ki,jkl->il",
                    lr_contract,
                    tensor_core,
                    drm_core,
                    optimize="optimal",
                )
            yield lr_contract[:, self.rank_min[mu] : self.rank_max[mu]]

    @handle_transpose
    def sketch_dense(self, tensor: DenseTensor) -> ArrayGenerator:
        n_dims = len(self.shape)
        partial_contraction = self.cores[0].reshape(-1, self.cores[0].shape[-1])
        yield partial_contraction.T
        for mu in range(1, n_dims - 1):
            core = self.cores[mu]
            partial_contraction = np.einsum(
                "ij,jkl->ikl", partial_contraction, core
            )
            partial_contraction = partial_contraction.reshape(
                -1, partial_contraction.shape[-1]
            )
            yield partial_contraction.T

    @handle_transpose
    def sketch_tucker(self, tensor: TuckerTensor) -> ArrayGenerator:
        n_dims = len(self.shape)
        partial_contraction = np.einsum(
            "ijk,jl->ilk", self.cores[0], tensor.factors[0].T
        )
        partial_contraction = partial_contraction.reshape(
            tensor.rank[0], self.rank[0]
        )
        yield partial_contraction

        for mu in range(1, n_dims - 1):
            core_reduced = np.einsum(
                "jkl,km->jml", self.cores[mu], tensor.factors[mu].T
            )
            partial_contraction = np.einsum(
                "ij,jml->iml", partial_contraction, core_reduced
            )
            partial_contraction = partial_contraction.reshape(
                -1, partial_contraction.shape[-1]
            )
            yield partial_contraction
