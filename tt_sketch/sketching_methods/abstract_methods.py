from abc import ABC, abstractmethod
from tt_sketch.tensor import TensorTrain, SparseTensor, DenseTensor, CPTensor
from tt_sketch.utils import ArrayGenerator
from typing import Tuple
from tt_sketch.drm_base import DRM


class CansketchTT(DRM, ABC):
    @abstractmethod
    def sketch_tt(self, tensor: TensorTrain) -> ArrayGenerator:
        r"""List of contractions of form :math:`Y_\mu^\top T_{\leq\mu}` where
        :math:`X_\mu` is the DRM, and :math"`T_{\leq\mu}` the
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
        r"""Return list of dense sketching matrices. Of shape
        ``(np.prod(tensor.shape[ :mu+1]), rank[mu])``"""


class CansketchCP(DRM, ABC):
    @abstractmethod
    def sketch_cp(self, tensor: CPTensor) -> ArrayGenerator:
        r"""List of contractions of form :math:`Y_\mu^\top T_{\leq\mu}` where
        :math:`X_\mu` is DRM, and :math"`T_{\leq\mu}` the
        contraction of the first :math:`\mu` cores of ``tensor``.

        Returns array of shape ``(tensor.rank[mu], drm.rank[mu])``"""
