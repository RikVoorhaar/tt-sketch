"""Implements the TT-GMRES algorithm for solving linear systems in the 
TT-format, as described in Dolgov, arXiv:1206.5512, but with rounding step
optionally replaced by sketching."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import logging
from math import ceil
from time import perf_counter
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import numpy as np
import numpy.typing as npt
import scipy.linalg

from tt_sketch.sketch import orthogonal_sketch, stream_sketch
from tt_sketch.tensor import Tensor, TensorSum, TensorTrain
from tt_sketch.utils import (
    ArrayList,
    TTRank,
    dematricize,
    matricize,
    process_tt_rank,
)


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

    def __mul__(self, other: float) -> MPO:
        new_cores = self.cores
        new_cores[0] = new_cores[0] * other
        return self.__class__(new_cores)


class TTPrecond(TTLinearMap):
    """TTLinearMap that acts by multiplying by the inverse of a matrix on a
    specified mode.

    The inverse is computed from the precomputed QR factorization of the matrix.
    """

    def __init__(self, A, shape, mode=0):
        self.A = A
        self.Q, self.R = np.linalg.qr(A)
        self.mode = mode
        self.in_shape = shape
        self.out_shape = shape

    def backward_call(self, other: TensorTrain) -> TensorTrain:
        new_cores = deepcopy(other.cores)
        C = new_cores[self.mode]
        C_mat = matricize(C, mode=1, mat_shape=True)
        sol = scipy.linalg.solve_triangular(self.R, (self.Q.T) @ C_mat)
        new_cores[self.mode] = dematricize(sol, mode=1, shape=C.shape)
        return TensorTrain(new_cores)

    def forward_call(self, other: TensorTrain) -> TensorTrain:
        new_cores = deepcopy(other.cores)
        C = new_cores[self.mode]
        C_mat = matricize(C, mode=1, mat_shape=True)
        new_cores[self.mode] = dematricize(
            self.A @ C_mat, mode=1, shape=C.shape
        )
        return TensorTrain(new_cores)

    __call__ = backward_call


class TTLinearMapSum:
    """Linear map that eats a TT and returns a sum of TTs.

    This is essentially just a container for a list of ``TTLinearMap`` objects.
    """

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    linear_maps: List[TTLinearMap]

    def __init__(self, linear_maps: List[TTLinearMap]) -> None:
        self.linear_maps = linear_maps
        if len(linear_maps) == 0:
            raise ValueError("linear_maps cannot be empty")
        self.in_shape = linear_maps[0].in_shape
        self.out_shape = linear_maps[0].out_shape
        for linear_map in linear_maps[1:]:
            if linear_map.in_shape != self.in_shape:
                raise ValueError("in_shape mismatch")
            if linear_map.out_shape != self.out_shape:
                raise ValueError("out_shape mismatch")

    def __call__(
        self, input_tensor: Union[TensorTrain, TensorSum[TensorTrain]]
    ) -> TensorSum[TensorTrain]:
        if isinstance(input_tensor, TensorTrain):
            tensor_list = [input_tensor]
        else:
            tensor_list = input_tensor.tensors

        output_list = []
        for linear_map in self.linear_maps:
            for tensor in tensor_list:
                output_list.append(linear_map(tensor))

        return TensorSum(output_list)


# def tt_weighted_sum_sketched(
#     x0: TensorTrain,
#     coeffs: npt.NDArray,
#     tt_list: List[TensorTrain],
#     tolerance: float,
#     max_rank: Tuple[int, ...],
#     round: bool = False,
# ):
#     """Sketched weighted sum of tensor trains."""
#     x_sum = TensorSum([x0])
#     for coeff, tt in zip(coeffs, tt_list):
#         x_sum += coeff * tt
#     x = round_tt_sum(x_sum, tolerance, max_rank, False)
#     return x


# def tt_weighted_sum_exact(
#     x0: TensorTrain,
#     coeffs: npt.NDArray,
#     tt_list: List[TensorTrain],
#     tolerance: float,
#     max_rank: Tuple[int, ...],
# ):
#     """Weighted sum of tensor trains rounded to ``max_rank``"""
#     x = x0
#     for coeff, tt in zip(coeffs, tt_list):
#         x = x.add(coeff * tt)
#     x = x.round(tolerance, max_rank)

#     return x


# def tt_weighted_sum(
#     x0: TensorTrain,
#     coeffs: npt.NDArray,
#     tt_list: List[TensorTrain],
#     tolerance: float,
#     max_rank: Tuple[int, ...],
#     exact: bool = False,
#     oversample: int = 5,
# ):
#     x_sum = TensorSum([x0])
#     for coeff, tt in zip(coeffs, tt_list):
#         x_sum += coeff * tt
#     x = round_tt_sum(x_sum, max_rank, tolerance, exact, oversample)
#     return x


ROUNDING_MODE = Literal["exact", "pairwise", "sketch", "orth_sketch", None]


def round_tt_sum(
    tt_sum: TensorSum[TensorTrain],
    max_rank: TTRank,
    eps: Optional[float] = None,
    method: ROUNDING_MODE = "sketch",
    oversample_factor: float = 2,
) -> TensorTrain:
    """Round a sum of tensor trains to a given rank.

    method can be one of:
    - "exact": Add all TTs to one big TT and round it using TT-SVD
    - "pairwise": Add each TT to the next one and round them separately
    - "sketch": Use streaming sketch for rounding
    - "orth_sketch": Use orthogonal sketch for rounding.
    - ``None``: Do not round (for debugging purposes mostly).
    """
    if method == "exact":
        summands = tt_sum.tensors
        tt = summands[0]
        for summand in summands[1:]:
            tt = tt.add(summand)
        return tt.round(eps, max_rank)
    elif method == "pairwise":
        tt = tt_sum.tensors[0]
        for t in tt_sum.tensors[1:]:
            tt = tt.add(t).round(eps=eps, max_rank=max_rank)
    elif method == "sketch":
        left_rank = process_tt_rank(max_rank, tt_sum.shape, trim=True)
        right_rank = tuple(ceil(r * oversample_factor) for r in left_rank)

        tt = stream_sketch(
            tt_sum, left_rank=left_rank, right_rank=right_rank
        ).to_tt()
    elif method == "orth_sketch":
        left_rank = process_tt_rank(max_rank, tt_sum.shape, trim=True)
        right_rank = tuple(ceil(r * oversample_factor) for r in left_rank)

        tt = orthogonal_sketch(
            tt_sum, left_rank=left_rank, right_rank=right_rank
        )
    elif method is None:
        return tt_sum  # type: ignore
    else:
        raise ValueError(f"Unknown rounding method: {method}")

    return tt


def tt_sum_gmres(
    A: TTLinearMapSum,
    b: TensorTrain,
    max_rank: TTRank,
    precond: Optional[TTPrecond] = None,
    final_round_rank: Optional[TTRank] = None,
    x0: Optional[TensorTrain] = None,
    tolerance: float = 1e-6,
    maxiter: int = 100,
    symmetric: bool = False,
    rounding_method: ROUNDING_MODE = "pairwise",
    rounding_method_final: Optional[ROUNDING_MODE] = None,
    save_basis: bool = False,
    verbose: bool = False,
) -> Tuple[TensorTrain, Dict[str, List]]:
    """
    GMRES solver for TTLinearMapSum.

    The TTLinearMapSum takes as input a TT and returns a sum of TTs. This means
    additional rounding steps are required in comparison to a version of
    TT-GMRES where the output of the linear map is a TT.
    """
    if final_round_rank is None:
        final_round_rank = max_rank
    if rounding_method_final is None:
        rounding_method_final = rounding_method
    if A.out_shape != b.shape:
        raise ValueError("Output shape of linear map doesn't match RHS")
    if x0 is not None and x0.shape != A.in_shape:
        raise ValueError("Input shape of liner map doesn't match initial value")
    if A.out_shape != A.in_shape:
        raise ValueError("TT-GMRES only works for automorphisms")

    max_rank = process_tt_rank(max_rank, A.in_shape, trim=True)
    if x0 is None:
        # TODO: check whether init with zero or random init is better
        # x0 = TensorTrain.random(shape=A.in_shape, rank=max_rank)
        x0 = TensorTrain.zero(shape=A.in_shape, rank=1)

    def apply_A_pr(x: TensorTrain) -> TensorSum[TensorTrain]:
        res = A(x)
        if precond is not None:
            res = TensorSum([precond(r) for r in res.tensors])
        return res

    b_pr = precond(b) if precond is not None else b

    b_norm = b.norm()
    initial_time = perf_counter()
    residual = b_pr - apply_A_pr(x0)
    residual_rounded = round_tt_sum(
        residual, max_rank=max_rank, method=rounding_method  # type: ignore
    )
    residual_norm = residual_rounded.norm()
    beta = residual_norm
    nu_list: List[TensorTrain] = [residual_rounded / beta]
    H_matrix = np.zeros((maxiter + 1, maxiter))

    history: Dict[str, Any] = defaultdict(list)
    history["w_norm"].append(nu_list[-1].norm())
    history["rank"].append(residual_rounded.rank)
    history["residual_norm"].append(residual_norm / b_norm)
    history["step_time"].append(perf_counter() - initial_time)

    for j in range(maxiter):
        current_time = perf_counter()
        delta = tolerance / (residual_norm / beta)
        if verbose:
            logging.info(
                f"Iteration {j + 1}/{maxiter}, residual norm: {residual_norm / b_norm:.4e}"
            )
        w_sum = apply_A_pr(nu_list[-1])
        w_rounded = round_tt_sum(
            w_sum, eps=delta, max_rank=max_rank, method=rounding_method
        )

        min_j = max(0, j - 2) if symmetric else 0
        for i in range(min_j, j + 1):
            H_matrix[i, j] = w_rounded.dot(nu_list[i])

        # Do Gram-Schmidt orthogonalization
        w_sum = (
            w_rounded
            - TensorSum(nu_list[min_j : j + 1]) * H_matrix[min_j : j + 1, j]
        )
        w_rounded = round_tt_sum(
            w_sum, eps=delta, max_rank=max_rank, method=rounding_method
        )
        H_matrix[j + 1, j] = w_rounded.norm()
        nu_list.append(w_rounded / H_matrix[j + 1, j])
        history["step_time"].append(perf_counter() - current_time)

        # Compute residual norm
        H_red = H_matrix[: j + 2, : j + 1]
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, (residual_norm,), _, _ = np.linalg.lstsq(H_red, e1, rcond=None)
        history["step_time_with_res_norm"].append(perf_counter() - current_time)
        history["residual_norm"].append(np.sqrt(residual_norm) / b_norm)
        history["rank"].append(w_rounded.rank)
        history["w_norm"].append(H_matrix[j + 1, j])
        history["delta"].append(delta)

        if residual_norm / b_norm < tolerance:
            break

    # Compute final result and round
    y = y[: j + 1]
    nu_list = nu_list[: j + 1]
    current_time = perf_counter()
    result = x0 + TensorSum(nu_list) * y
    result_rounded = round_tt_sum(
        result,
        eps=None,
        max_rank=final_round_rank,
        method=rounding_method_final,
    )

    history["final_round_time"] = perf_counter() - current_time
    history["total_time"] = perf_counter() - initial_time
    if save_basis:
        history["H_matrix"] = H_matrix
        history["nu_list"] = nu_list
        history["y"] = y
    return result_rounded, history
