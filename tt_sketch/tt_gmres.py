"""Implements the TT-GMRES algorithm for solving linear systems in the 
TT-format, as described in Dolgov, arXiv:1206.5512. """
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from tt_sketch.sketch import stream_sketch
from tt_sketch.tensor import Tensor, TensorSum, TensorTrain
from tt_sketch.utils import ArrayList, TTRank, process_tt_rank


class TTLinearMap(ABC):
    """Abstract class for linear maps in the TT-format."""

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]

    @abstractmethod
    def __call__(self, other: TensorTrain) -> TensorTrain:
        ...


class MPO(Tensor, TTLinearMap):
    """MPO with order 4 tensor cores of shape
    ``(rank[mu-1],in_shape[mu],out_shape[mu],rank[mu])``

    Used as linear map in the tensor train format"""

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    rank: Tuple[int, ...]
    shape: Tuple[int, ...]
    cores: ArrayList

    def __init__(self, cores: ArrayList) -> None:
        self.cores = cores
        self.in_shape = tuple(C.shape[1] for C in cores)
        self.out_shape = tuple(C.shape[2] for C in cores)
        self.rank = tuple(C.shape[0] for C in cores[1:])
        self.shape = tuple(
            s1 * s2 for s1, s2 in zip(self.in_shape, self.out_shape)
        )

    @property
    def size(self) -> int:
        return sum(C.size for C in self.cores)

    @property
    def T(self) -> MPO:
        """Transposition here is that of a linear map, this is different from
        other tensors."""
        new_cores = [C.transpose((0, 2, 1, 3)) for C in self.cores]
        return self.__class__(new_cores)

    def to_tt(self) -> TensorTrain:
        new_cores = [
            C.reshape(C.shape[0], C.shape[1] * C.shape[2], C.shape[3])
            for C in self.cores
        ]
        return TensorTrain(new_cores)

    def to_numpy(mpo) -> npt.NDArray:
        """Contract to dense array of shape
        ``(in_shape[0], out_shape[0], ..., in_shape[d-1], outs_shape[d-1])``"""

        res = mpo.cores[0]
        res = res.reshape(res.shape[1:])
        for C in mpo.cores[1:]:
            res = np.einsum("...i,ijkl->...jkl", res, C)
        res = res.reshape(res.shape[:-1])
        return res

    def __call__(self, other: TensorTrain) -> TensorTrain:
        new_cores = []
        for M, C in zip(self.cores, other.cores):
            MC = np.einsum("ijkl,ajb->iaklb", M, C)
            MC = MC.reshape(
                MC.shape[0] * MC.shape[1],
                MC.shape[2],
                MC.shape[3] * MC.shape[4],
            )
            new_cores.append(MC)
        return TensorTrain(new_cores)

    @classmethod
    def random(
        cls,
        rank: TTRank,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
    ) -> MPO:
        prod_shape = tuple(s1 * s2 for s1, s2 in zip(in_shape, out_shape))
        rank = process_tt_rank(rank, prod_shape, trim=True)
        cores = []
        for r1, s1, s2, r2 in zip(
            (1,) + rank, in_shape, out_shape, rank + (1,)
        ):
            C = np.random.normal(size=(r1, s1, s2, r2))
            C += C.transpose(0, 2, 1, 3).reshape(C.shape)  # symmetrize
            C = C * np.sqrt(s1 * s2) / np.linalg.norm(C)
            cores.append(C)
        return cls(cores)

    @classmethod
    def eye(cls, shape) -> MPO:
        cores = []
        for s in shape:
            C = np.eye(s, s)
            C = C.reshape(1, C.shape[0], C.shape[1], 1)
            cores.append(C)
        return cls(cores)

    def __mul__(self, other: float) -> Tensor:
        new_cores = self.cores
        new_cores[0] = new_cores[0] * other
        return self.__class__(new_cores)


def tt_sum_round_orthog(
    X: TensorSum, epsilon: float, max_rank: Tuple[int, ...]
) -> TensorTrain:
    """Rounds and orthogonalizes a sum of tensor trains."""
    max_rank_trimmed = process_tt_rank(max_rank, X.shape, trim=True)
    left_rank = max_rank_trimmed
    right_rank = tuple(r * 2 for r in max_rank_trimmed)
    tt = stream_sketch(X, left_rank=left_rank, right_rank=right_rank).to_tt()
    tt = tt.round(eps=epsilon, max_rank=max_rank_trimmed)
    tt = tt.orthogonalize()
    return tt


def tt_weighted_sum_sketched(
    x0: TensorTrain,
    coeffs: npt.NDArray,
    tt_list: List[TensorTrain],
    tolerance: float,
    max_rank: Tuple[int, ...],
):
    """Sketched weighted sum of tensor trains."""
    x_sum = TensorSum([x0])
    for coeff, tt in zip(coeffs, tt_list):
        x_sum += coeff * tt
    x = tt_sum_round_orthog(x_sum, tolerance, max_rank)
    return x


def tt_weighted_sum_exact(
    x0: TensorTrain,
    coeffs: npt.NDArray,
    tt_list: List[TensorTrain],
    tolerance: float,
    max_rank: Tuple[int, ...],
):
    """Weighted sum of tensor trains rounded to ``max_rank``"""
    x = x0
    for coeff, tt in zip(coeffs, tt_list):
        x = x.add(coeff * tt)
    x = x.round(tolerance, max_rank)

    return x


tt_weighted_sum = tt_weighted_sum_sketched


def tt_gmres(
    A: TTLinearMap,
    b: TensorTrain,
    max_rank: TTRank,
    x0: Optional[TensorTrain] = None,
    tolerance: float = 1e-6,
    maxiter: int = 100,
) -> Tuple[TensorTrain, Dict[str, List]]:
    """GMRES solver for TT linear map."""
    if A.out_shape != b.shape:
        raise ValueError("Output shape of MPO doesn't match RHS")
    if x0 is not None and x0.shape != A.in_shape:
        raise ValueError("Input shape of MPO doesn't match initial value")
    if A.out_shape != A.in_shape:
        raise ValueError("TT-GMRES only works for Hermitian tensors")

    max_rank = process_tt_rank(max_rank, A.in_shape, trim=True)
    if x0 is None:
        # TODO: check whether init with zero or random init is better
        x0 = TensorTrain.random(shape=A.in_shape, rank=max_rank)
        # x0 = TensorTrain.zero(shape=A.in_shape, rank=max_rank)

    residual = b.add(A(x0) * (-1)).round(eps=tolerance, max_rank=max_rank)
    residual_norm = residual.norm()
    history: Dict[str, List] = defaultdict(list)
    history["residual_norm"].append(residual_norm)
    history["rank"].append(residual.rank)
    b_norm = b.norm()
    beta = residual_norm
    nu_list: List[TensorTrain] = [residual / beta]
    history["w_norm"].append(nu_list[-1].norm())

    H_matrix = np.zeros((maxiter + 1, maxiter))

    for j in range(maxiter):
        delta = tolerance / (residual_norm / beta)
        w = A(nu_list[-1]).round(eps=delta, max_rank=max_rank)

        for i in range(j + 1):
            H_matrix[i, j] = w.dot(nu_list[i])

        w = tt_weighted_sum(
            w, -H_matrix[: j + 1, j], nu_list[: j + 1], tolerance, max_rank
        )
        H_matrix[j + 1, j] = w.norm()
        nu_list.append(w / H_matrix[j + 1, j])

        H_red = H_matrix[: j + 2, : j + 1]
        e1 = np.zeros(j + 2)
        e1[0] = 1
        y, (residual_norm,), _, _ = np.linalg.lstsq(
            H_red, beta * e1, rcond=None
        )
        history["residual_norm"].append(residual_norm)
        history["rank"].append(w.rank)
        history["w_norm"].append(H_matrix[j + 1, j])

        if residual_norm / b_norm < tolerance:
            break

    x = tt_weighted_sum(x0, y[: j + 1], nu_list[: j + 1], tolerance, max_rank)

    return x, history
