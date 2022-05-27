from abc import ABC, abstractmethod
from typing import Tuple

from tt_sketch.drm_base import DRM
from tt_sketch.tensor import (
    CPTensor,
    DenseTensor,
    SparseTensor,
    TensorTrain,
    TuckerTensor,
)
from tt_sketch.utils import ArrayGenerator


class CansketchTT(DRM, ABC):
    @abstractmethod
    def sketch_tt(self, tensor: TensorTrain) -> ArrayGenerator:
        r"""List of contractions of form :math:`Y_\mu^\top\mathcal{T}^{\leq\mu}` where
        :math:`Y_\mu` is the DRM, and :math:`\mathcal T^{\leq\mu}` the
        contraction of the first :math:`\mu` cores of ``tensor``.

        Returns array of shape ``(tensor.rank[mu], drm.rank[mu])``"""


class CansketchSparse(DRM, ABC):
    rank: Tuple[int, ...]

    @abstractmethod
    def sketch_sparse(self, tensor: SparseTensor) -> ArrayGenerator:
        """Computes list of sketching matrices sampled into a vector using the
        indices of ``tensor`` for each unfolding. Shape of each vector is
        ``v[mu] = (rank[mu], tensor.nnz)``. This way the contraction between
        ``tensor`` and the sketching matrix is of form ``np.dot(tensor.entries,
        v[mu])``"""


class CansketchDense(DRM, ABC):
    @abstractmethod
    def sketch_dense(self, tensor: DenseTensor) -> ArrayGenerator:
        r"""Return list of dense DRMs. Of shape
        ``(np.prod(tensor.shape[ :mu+1]), rank[mu])``"""


class CansketchCP(DRM, ABC):
    @abstractmethod
    def sketch_cp(self, tensor: CPTensor) -> ArrayGenerator:
        r"""List of contractions of form :math:`Y_\mu^\top\mathcal{T}^{\leq\mu}` where
        :math:`Y_\mu` is the DRM, and :math:`\mathcal T^{\leq\mu}` the
        contraction of the first :math:`\mu` cores of ``tensor``.

        Returns array of shape ``(tensor.rank[mu], drm.rank[mu])``"""


class CanSketchTucker(DRM, ABC):
    @abstractmethod
    def sketch_tucker(self, tensor: TuckerTensor) -> ArrayGenerator:
        r"""
        List of contractions of form :math:`Y_\mu^\top(U_1\otimes\cdots\otimes U_\mu)`
        where ``Y_\mu`` is the DRM, and :math:`U_\mu` denotes the factors of the
        Tucker decomposition.

        Returns array of shape ``(np.prod(tensor.rank[:mu]), drm.rank[mu])``
        """
