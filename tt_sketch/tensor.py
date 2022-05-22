"""Implements Tensor classes"""
from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple, Union, Type

import numpy as np
import numpy.typing as npt

# from tt_sketch.tt_svd import tt_svd
from tt_sketch.utils import (
    ArrayList,
    TTRank,
    process_tt_rank,
    right_mul_pinv,
    left_mul_pinv,
)
from tt_sketch.sketch_container import SketchContainer
from tt_sketch.drm_base import DRM


class Tensor(ABC):
    shape: Tuple[int, ...]

    @abstractproperty
    def T(self) -> Tensor:
        pass

    @abstractmethod
    def to_numpy(self) -> npt.NDArray[np.float64]:
        pass

    def mse_error(self, other: Union[Tensor, npt.NDArray]) -> float:
        """Computes the MSE error"""
        if isinstance(other, Tensor):
            other = other.to_numpy()
        square_error = np.linalg.norm(self.to_numpy() - other) ** 2
        return square_error / np.prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dense(self) -> DenseTensor:
        dense_tensor = self.to_numpy()
        return DenseTensor(self.shape, dense_tensor)

    def __add__(self, other) -> TensorSum:
        if isinstance(other, TensorSum):
            if not isinstance(self, TensorSum):
                return TensorSum([self] + other.tensors)
            else:
                return TensorSum(self.tensors + other.tensors)
        elif isinstance(self, TensorSum):
            return TensorSum(self.tensors + [other])
        else:
            return TensorSum([self, other])


@dataclass
class DenseTensor(Tensor):
    shape: Tuple[int, ...]
    data: npt.NDArray[np.float64]

    dtype = np.dtype("float64")

    def to_sparse(self) -> SparseTensor:
        X = self.data
        n_dims = len(X.shape)
        inds = np.indices(X.shape).reshape(n_dims, -1)
        entries = X.reshape(-1)
        return SparseTensor(X.shape, inds, entries)

    @property
    def T(self) -> DenseTensor:
        new_shape = self.shape[::-1]
        permutation = tuple(range(len(self.shape))[::-1])
        new_data = np.transpose(self.data, permutation)
        return self.__class__(new_shape, new_data)

    def __getitem__(self, key):
        new_data = self.data.__getitem__(key)
        new_shape = new_data.shape
        return self.__class__(new_shape, new_data)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self.data

    def __repr__(self) -> str:
        return f"<Dense tensor of shape {self.shape} at {hex(id(self))}>"


@dataclass
class SparseTensor(Tensor):
    shape: Tuple[int, ...]
    indices: npt.NDArray
    entries: npt.NDArray[np.float64]

    @property
    def T(self) -> SparseTensor:
        new_indices = self.indices[::-1]
        new_shape = self.shape[::-1]
        return self.__class__(new_shape, new_indices, self.entries)

    @property
    def nnz(self) -> int:
        return len(self.entries)

    def split(self, n_summands: int) -> TensorSum:
        block_size = self.nnz // n_summands
        summands: List[Tensor] = []
        for i in range(n_summands):
            if i < n_summands - 1:
                summand_slice = slice(i * block_size, (i + 1) * block_size)
            else:
                summand_slice = slice(i * block_size, self.nnz)

            summands.append(
                SparseTensor(
                    self.shape,
                    self.indices[:, summand_slice],
                    self.entries[summand_slice],
                )
            )
        return TensorSum(summands)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        X = np.zeros(self.shape)
        X[tuple(self.indices)] = self.entries
        return X

    def norm(self) -> float:
        return float(np.linalg.norm(self.entries))

    def __repr__(self) -> str:
        return (
            f"<Sparse tensor of shape {self.shape} with {self.nnz} non-zero"
            f" entries at {hex(id(self))}>"
        )


class TensorTrain(Tensor):
    shape: Tuple[int, ...]
    rank: Tuple[int, ...]
    cores: ArrayList

    def __init__(self, cores: ArrayList) -> None:
        self.cores = cores
        self.shape = tuple(C.shape[1] for C in cores)
        self.rank = tuple(C.shape[0] for C in cores[1:])

    @property
    def T(self) -> TensorTrain:
        new_cores = [np.transpose(C, (2, 1, 0)) for C in self.cores[::-1]]
        return self.__class__(new_cores)

    def to_numpy(self) -> npt.NDArray:
        dense_tensor = self.cores[0]
        dense_tensor = dense_tensor.reshape(dense_tensor.shape[1:])
        for C in self.cores[1:]:
            dense_tensor = np.einsum("...j,jkl->...kl", dense_tensor, C)
        dense_tensor = dense_tensor.reshape(dense_tensor.shape[:-1])
        return dense_tensor

    @classmethod
    def random(
        cls,
        shape: Tuple[int, ...],
        rank: TTRank,
        seed: Optional[int] = None,
        orthog: bool = False,
    ) -> TensorTrain:
        """Generate random orthogonal tensor train cores"""
        if seed is not None:
            np.random.seed(seed)
        d = len(shape)
        rank = process_tt_rank(rank, shape, trim=False)
        rank_augmented = (1,) + tuple(rank) + (1,)
        cores = []
        for i in range(d):
            r1 = rank_augmented[i]
            r2 = rank_augmented[i + 1]
            n = shape[i]
            core = np.random.normal(size=(r1 * n, r2))
            if orthog and i < d - 1:
                core, _ = np.linalg.qr(core, mode="reduced")
            else:
                core /= np.sqrt(r1 * n * r2)
            core = core.reshape(r1, n, r2)
            cores.append(core)

        return cls(cores)

    @classmethod
    def zero(cls, shape: Tuple[int, ...], rank: TTRank) -> TensorTrain:
        d = len(shape)

        rank = process_tt_rank(rank, shape, trim=False)
        cores = []
        for (r1, d, r2) in zip((1,) + rank, shape, rank + (1,)):
            cores.append(np.zeros((r1, d, r2)))
        return cls(cores)

    def partial_dense(self, dir: str = "lr") -> ArrayList:
        """Do partial contractions to dense tensor; ``X[0].X[1]...X[mu]``"""
        if dir == "lr":
            partial_cores = [self.cores[0].reshape(-1, self.cores[0].shape[-1])]
            for core in self.cores[1:-1]:
                new_core = np.einsum("ij,jkl->ikl", partial_cores[-1], core)
                new_core = new_core.reshape(-1, new_core.shape[-1])
                partial_cores.append(new_core)
        elif dir == "rl":
            partial_cores = [
                self.cores[-1].reshape(self.cores[-1].shape[0], -1)
            ]
            for core in self.cores[-2:0:-1]:
                new_core = np.einsum("ijk,kl->ijl", core, partial_cores[-1])
                new_core = new_core.reshape(new_core.shape[0], -1)
                partial_cores.append(new_core)
        return partial_cores

    def __getitem__(self, index: int) -> npt.NDArray:
        return self.cores[index]

    def __setitem__(self, index: int, data: npt.NDArray) -> None:
        self.cores[index] = data

    def gather(
        self,
        idx: Union[npt.NDArray[np.int64], Tuple[npt.NDArray[np.int64], ...]],
    ) -> npt.NDArray:
        """Gather entries of dense tensor according to indices.

        For each row of `idx` this returns one number. This number is obtained
        by multiplying the slices of each core corresponding to each index (in
        a left-to-right fashion).
        """
        if not isinstance(idx, np.ndarray):
            idx_array = np.stack(idx)
        else:
            idx_array = np.array(idx)
        N = idx_array.shape[1]
        result = np.take(
            self[0].reshape(self[0].shape[1:]), idx_array[0], axis=0
        )
        for i in range(1, self.ndim):
            r = self[i].shape[2]
            next_step = np.zeros((N, r))
            for j in range(self.shape[i]):
                idx_mask = np.where(idx_array[i] == j)
                mat = self[i][:, j, :]
                next_step[idx_mask] = result[idx_mask] @ mat
            result = next_step
        return result.reshape(-1)

    def round(
        self,
        eps: Optional[float] = None,
        max_rank: Optional[TTRank] = None,
    ) -> TensorTrain:
        """Standard TT-SVD rounding scheme.

        First left orthogonalize in LR sweep, then apply SVD-based rounding in
        a RL sweep. Leaves the tensor right-orthogonalized."""
        tt = self.orthogonalize()  # left-orthogonalize
        if eps is None:
            eps = 0
        if max_rank is None:
            max_rank = tt.rank
        max_rank = process_tt_rank(max_rank, tt.shape, trim=True)

        new_cores = []
        US_trunc: npt.NDArray
        for mu, C in list(enumerate(tt.cores))[::-1]:
            if mu < tt.ndim - 1:
                C = np.einsum("ijk,kl->ijl", C, US_trunc)
            if mu > 0:
                C_mat = C.reshape(C.shape[0], C.shape[1] * C.shape[2])
                U, S, Vt = np.linalg.svd(C_mat)
                r = max(1, min(np.sum(S > S[0] * eps), max_rank[mu - 1]))
                US_trunc = U[:, :r] @ np.diag(S[:r])
                Vt_trunc = Vt[:r, :]
                new_cores.append(Vt_trunc.reshape(r, C.shape[1], C.shape[2]))
            else:
                new_cores.append(C)

        return self.__class__(new_cores[::-1])

    def __mul__(self, other: Union[float, int]) -> TensorTrain:
        new_cores = deepcopy(self.cores)
        # Absorb number into last core, since if we do orthogonalize, we left
        # orthogonalize.
        new_cores[-1] = new_cores[-1] * other
        return self.__class__(new_cores)

    __rmul__ = __mul__

    def __truediv__(self, other: Union[float, int]):
        return self.__mul__(1 / other)

    def add(self, other: TensorTrain) -> TensorTrain:
        """Add two ``TensorTrain`` by taking direct sums of cores.
        Note, this does not overload the addition operator ``+``; the addition
        operator instead returns a lazy ``TensorSum`` object.
        """
        new_cores = [np.concatenate((self.cores[0], other.cores[0]), axis=2)]
        for C1, C2 in zip(self.cores[1:-1], other.cores[1:-1]):
            r1, d, r2 = C1.shape
            r3, _, r4 = C2.shape
            zeros1 = np.zeros((r1, d, r4))
            zeros2 = np.zeros((r3, d, r2))
            row1 = np.concatenate((C1, zeros1), axis=2)
            row2 = np.concatenate((zeros2, C2), axis=2)
            new_cores.append(np.concatenate((row1, row2), axis=0))
        new_cores.append(
            np.concatenate((self.cores[-1], other.cores[-1]), axis=0)
        )
        new_tt = self.__class__(new_cores)
        return new_tt

    def dot(self, other: TensorTrain) -> float:
        """Compute the dot product of two tensor trains with the same shape.

        Result is computed in a left-to-right sweep."""

        result = np.einsum("ijk,ljm->km", self.cores[0], other.cores[0])
        for core1, core2 in zip(self.cores[1:], other.cores[1:]):
            # optimize is essential here, reduces complexity from r^4*n to r^3*n
            result = np.einsum(
                "ij,ika,jkb->ab", result, core1, core2, optimize="optimal"
            )
        return np.sum(result)

    def norm(self) -> float:
        return np.sqrt(self.dot(self))

    def orthogonalize(self) -> TensorTrain:
        """Do QR sweep to left-orthogonalize"""
        new_cores = []
        R: npt.NDArray
        for mu, C in enumerate(self.cores):
            if mu > 0:
                C = np.einsum("ij,jkl->ikl", R, C)
            if mu < self.ndim - 1:
                C_mat = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
                Q, R = np.linalg.qr(C_mat)
                new_cores.append(Q.reshape(C.shape[0], C.shape[1], -1))
            else:
                new_cores.append(C)
        return self.__class__(new_cores)

    def __repr__(self) -> str:
        return (
            f"<Tensor train of shape {self.shape} with rank {self.rank}"
            f" at {hex(id(self))}>"
        )


def diff_tt_sparse(tt: TensorTrain, sparse_tensor: SparseTensor) -> float:
    """
    Compute the (Frobenius) norm of the difference of a TT and a sparse tensor
    """
    norm_squared = (
        tt.norm() ** 2
        + sparse_tensor.norm() ** 2
        - 2 * np.dot(tt.gather(sparse_tensor.indices), sparse_tensor.entries)
    )
    if norm_squared < 0:
        return 0
    return norm_squared**0.5


class SketchedTensorTrain(Tensor):
    sketch_: SketchContainer
    left_rank: Tuple[int, ...]
    right_rank: Tuple[int, ...]

    @property
    def Psi_cores(self) -> ArrayList:
        return self.sketch_.Psi_cores

    @property
    def Omega_mats(self) -> ArrayList:
        return self.sketch_.Omega_mats

    def __init__(
        self,
        sketch: SketchContainer,
    ):
        self.sketch_ = sketch
        self.shape = sketch.shape
        self.left_rank = sketch.left_rank
        self.right_rank = sketch.right_rank

    @classmethod
    def zero(cls, shape, left_rank, right_rank) -> SketchedTensorTrain:
        """Initialize a new ``SketchedTensorTrain`` with zeros as ``Psi_cores``
        ``Omega_mats``."""
        sketch = SketchContainer.zero(shape, left_rank, right_rank)
        return cls(sketch)

    def C_cores(self, direction="auto") -> ArrayList:
        return assemble_sketched_tt(self.sketch_, direction=direction)

    @property
    def T(self) -> SketchedTensorTrain:
        new_sketch = self.sketch_.T
        return self.__class__(new_sketch)

    def to_tt(self) -> TensorTrain:
        return TensorTrain(self.C_cores())

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self.to_tt().to_numpy()

    def __repr__(self) -> str:
        return (
            f"<Sketched tensor train of shape {self.shape} with left-rank "
            f"{self.left_rank} and right-rank {self.right_rank} "
            f"at {hex(id(self))}>"
        )


# TODO: make left and right version. Pick best by default
def assemble_sketched_tt(
    sketch: SketchContainer,
    eps: float = 1e-15,
    direction="auto",
) -> ArrayList:
    tt_cores = []
    if direction == "auto":
        left_rank_bigger = np.all(
            np.array(sketch.left_rank) > np.array(sketch.right_rank)
        )
        direction = "left" if left_rank_bigger else "right"

    if direction == "right":
        for Psi, Omega in zip(sketch.Psi_cores[:-1], sketch.Omega_mats):
            Psi_shape = Psi.shape
            Psi_mat = Psi.reshape(Psi_shape[0] * Psi_shape[1], Psi_shape[2])
            try:
                Psi_Omega_pinv = right_mul_pinv(Psi_mat, Omega)
            except ValueError:
                print(Psi.shape, Omega.shape)
                raise
            core = Psi_Omega_pinv.reshape(
                Psi_shape[0], Psi_shape[1], Omega.shape[0]
            )
            tt_cores.append(core)
        tt_cores.append(sketch.Psi_cores[-1])
    elif direction == "left":
        tt_cores.append(sketch.Psi_cores[0])
        for Psi, Omega in zip(sketch.Psi_cores[1:], sketch.Omega_mats):
            Psi_shape = Psi.shape
            Psi_mat = Psi.reshape(Psi_shape[0], Psi_shape[1] * Psi_shape[2])
            try:
                Omega_pinv_Psi = left_mul_pinv(Omega, Psi_mat)
            except ValueError:
                print(Psi.shape, Omega.shape)
                raise
            core = Omega_pinv_Psi.reshape(
                Omega.shape[1], Psi_shape[1], Psi_shape[2]
            )
            tt_cores.append(core)
    else:
        raise ValueError(f"Unknown direction {direction}")

    return tt_cores


class TensorSum(Tensor):
    """Container for sums of tensors"""

    shape: Tuple[int, ...]
    tensors: List[Tensor]

    def __init__(self, tensors: List[Tensor], shape=None) -> None:
        if shape is None:
            shape = tensors[0].shape
        self.shape = shape
        self.tensors = tensors

    @property
    def T(self) -> TensorSum:
        new_tensors = [X.T for X in self.tensors]
        return self.__class__(new_tensors)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        s = np.zeros(self.shape)
        for X in self.tensors:
            s += X.to_numpy()
        return s

    def __add__(self, other: Tensor) -> TensorSum:
        if isinstance(other, TensorSum):
            return self.__class__(self.tensors + other.tensors)
        else:
            return self.__class__(self.tensors + [other])

    def __iadd__(self, other: Tensor) -> TensorSum:
        if isinstance(other, TensorSum):
            self.tensors.extend(other.tensors)
        else:
            self.tensors.append(other)
        return self

    def __repr__(self) -> str:
        return (
            f"<Sum of {len(self.tensors)} tensors of shape {self.shape}"
            f" at {hex(id(self))}>"
        )


class CPTensor(Tensor):
    """Implements CP tensors.

    The cores are stored as list of shape ``(shape[i], rank)``."""

    shape: Tuple[int, ...]
    rank: int
    cores: ArrayList

    def __init__(self, cores: ArrayList) -> None:
        self.cores = cores
        self.rank = cores[0].shape[1]
        self.shape = tuple(C.shape[0] for C in cores)

    @property
    def T(self) -> CPTensor:
        new_cores = self.cores[::-1]
        return self.__class__(new_cores)

    def to_numpy(self) -> npt.NDArray:
        dense_tensor = self.cores[0]  # shape (n0,r)
        for C in self.cores[1:]:
            dense_tensor = np.einsum(
                "...j,ij->...ij", dense_tensor, C
            )  # shape (n0,...,nk,r)
        dense_tensor = np.sum(dense_tensor, axis=-1)
        return dense_tensor

    @classmethod
    def random(
        cls, shape: Tuple[int, ...], rank: int, seed: Optional[int] = None
    ) -> CPTensor:
        if seed is not None:
            np.random.seed(seed)
        d = len(shape)
        cores = []
        for i in range(d):
            core = np.random.normal(size=(shape[i], rank))
            core /= np.sqrt(shape[i])
            cores.append(core)

        return cls(cores)

    def __getitem__(self, index: int) -> npt.NDArray:
        return self.cores[index]

    def __setitem__(self, index: int, data: npt.NDArray) -> None:
        self.cores[index] = data

    def gather(self, idx: Tuple[npt.NDArray[np.int64], ...]) -> npt.NDArray:
        res = 1
        for C, id in zip(self.cores, idx):
            res *= C[id]
        return np.sum(res, axis=1)

    def __repr__(self) -> str:
        return (
            f"<CP tensor of shape {self.shape} and rank {self.rank} "
            f"at {hex(id(self))}>"
        )
