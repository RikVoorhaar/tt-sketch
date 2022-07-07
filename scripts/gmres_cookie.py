# %%
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.io
import scipy.linalg
from tqdm import tqdm
from tt_sketch.sketch import orthogonal_sketch, stream_sketch
from tt_sketch.tensor import Tensor, TensorSum, TensorTrain
from tt_sketch.tt_gmres import TTLinearMap, tt_weighted_sum
from tt_sketch.utils import (
    ArrayList,
    TTRank,
    dematricize,
    matricize,
    process_tt_rank,
    trim_ranks,
)


class TTLinearMapSum:
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


# %%


cookies_data = scipy.io.loadmat("data/cookies_matrices_2x2.mat")
b = cookies_data["b"].reshape(-1)
A_list = cookies_data["A"][0]


class TTLinearMapInverse(TTLinearMap):
    """TTLinearMap that acts by multiplying by the inverse of a matrix on a
    specified mode.

    The inverse is computed from the precomputed QR factorization of the matrix."""

    def __init__(self, A, shape, mode=0):
        self.A = A
        self.Q, self.R = np.linalg.qr(A)
        self.mode = mode
        self.in_shape = shape
        self.out_shape = shape

    def precond_call(self, other: TensorTrain) -> TensorTrain:
        new_cores = deepcopy(other.cores)
        C = new_cores[self.mode]
        C_mat = matricize(C, mode=1, mat_shape=True)
        sol = scipy.linalg.solve_triangular(self.R, (self.Q.T) @ C_mat)
        new_cores[self.mode] = dematricize(sol, mode=1, shape=C.shape)
        return TensorTrain(new_cores)

    __call__ = precond_call

class CookieMap(TTLinearMap):
    def __init__(
        self,
        A: npt.NDArray,
        mode: int,
        shape: Tuple[int, ...],
        coeffs: npt.NDArray,
        preconditioner: Optional[TTLinearMapInverse] = None,
    ) -> None:
        self.A = A
        self.preconditioner = preconditioner
        self.mode = mode
        self.in_shape = shape
        self.out_shape = shape
        self.coeffs = coeffs

    def cookie_call(self, other: TensorTrain) -> TensorTrain:
        new_cores = deepcopy(other.cores)
        new_cores[0] = np.einsum(
            "ijk,jl->ilk", new_cores[0], self.A, optimize="optimal"
        )
        new_cores[self.mode] = np.einsum(
            "ijk,j->ijk", new_cores[self.mode], self.coeffs, optimize="optimal"
        )
        tt = TensorTrain(new_cores)
        if self.preconditioner is not None:
            tt = self.preconditioner(tt)
        return tt

    __call__ = cookie_call


num_coeffs = 10
shape = (A_list[0].shape[0],) + (num_coeffs,) * (len(A_list) - 1)

A_precond_list = []
coeffs_list = []
for mu, A in enumerate(A_list):
    A = A.toarray()
    if mu == 0:
        coeffs = np.ones(A.shape[0])
    else:
        coeffs = -np.arange(num_coeffs, dtype=np.float64) + 1
    A_precond_list.append(A * np.mean(coeffs))
    coeffs_list.append(coeffs)

A_precond = np.sum(A_precond_list, axis=0)
precond_map = TTLinearMapInverse(A_precond, shape, mode=0)

cookie_maps = []

for mu, (A, coeffs) in enumerate(zip(A_list, coeffs_list)):
    cookie_maps.append(
        CookieMap(
            A.toarray(),
            mu,
            shape,
            coeffs,
            preconditioner=precond_map,
        )
    )

map_sum = TTLinearMapSum(cookie_maps)


tt = TensorTrain.random(shape, 50)

B_cores = [b.reshape(1, -1, 1)]
for n in shape[1:]:
    B_cores.append(np.ones((1, n, 1)))
B = TensorTrain(B_cores)
B_pr = precond_map(B)
# %%

mtt = map_sum(tt)
mtt = mtt + mtt
# ranks = np.arange(1, 10)
# norms = []
# for rank in tqdm(ranks):
#     stt = stream_sketch(mtt, left_rank=rank, right_rank=rank + 10)
#     norms.append(stt.error(mtt, relative=True))

# plt.plot(ranks, norms)
# plt.yscale("log")

from tt_sketch.tt_gmres import tt_weighted_sum_exact, tt_weighted_sum_sketched

summands = mtt.tensors


def do_wse():
    for _ in range(1):
        wse = tt_weighted_sum_exact(
            summands[0],
            summands[1:],
            np.ones(len(summands) - 1),
            tolerance=1e-8,
            max_rank=tt.rank,
        )


def do_wss():
    for _ in range(1):
        wss = tt_weighted_sum_sketched(
            summands[0],
            summands[1:],
            np.ones(len(summands) - 1),
            tolerance=1e-8,
            max_rank=tt.rank,
        )


# %time do_wse()
# %time do_wss()
# %%
def round_tt_sum(
    tt_sum: TensorSum[TensorTrain],
    eps=1e-8,
    max_rank: TTRank = None,
    exact: bool = False,
):
    if exact:
        summands = tt_sum.tensors
        tt = summands[0]
        for summand in summands[1:]:
            tt = tt.add(summand)
        return tt.round(eps, max_rank)
    else:
        left_rank = trim_ranks(tt_sum.shape, max_rank)
        right_rank = tuple(r + 5 for r in left_rank)

        tt = orthogonal_sketch(
            tt_sum, left_rank=left_rank, right_rank=right_rank
        )
    return tt


def tt_sum_gmres(
    A: TTLinearMapSum,
    b: TensorTrain,
    max_rank: TTRank,
    x0: Optional[TensorTrain] = None,
    tolerance: float = 1e-6,
    maxiter: int = 100,
) -> Tuple[TensorTrain, Dict[str, List]]:
    """GMRES solver for TT linear map."""
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
        x0 = TensorTrain.zero(shape=A.in_shape, rank=max_rank)

    residual = b - A(x0)
    residual = round_tt_sum(residual, max_rank=max_rank)
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
        w = A(nu_list[-1])  # .round(eps=delta, max_rank=max_rank)
        w = round_tt_sum(w, eps=delta, max_rank=max_rank)

        for i in range(j + 1):
            H_matrix[i, j] = w.dot(nu_list[i])

        w = tt_weighted_sum_sketched(
            w, -H_matrix[: j + 1, j], nu_list[: j + 1], tolerance, max_rank, round=False
        )
        # for coeff, nu in zip(H_matrix[: j + 1, j], nu_list[: j + 1]):
        #     w += coeff * nu
        H_matrix[j + 1, j] = w.norm()
        nu_list.append(w / H_matrix[j + 1, j])

        H_red = H_matrix[: j + 2, : j + 1]
        e1 = np.zeros(j + 2)
        e1[0] = 1
        y, (residual_norm,), _, _ = np.linalg.lstsq(
            H_red, beta * e1, rcond=None
        )
        history["residual_norm"].append(residual_norm / b_norm)
        history["rank"].append(w.rank)
        history["w_norm"].append(H_matrix[j + 1, j])
        print(history["residual_norm"][-1], history["rank"][-1])

        if residual_norm / b_norm < tolerance:
            break

    x = tt_weighted_sum_sketched(
        x0, y[: j + 1], nu_list[: j + 1], tolerance, max_rank, round=True
    )
    history["H_matrix"] = H_matrix
    return x, history


x0 = TensorTrain.zero(shape=map_sum.in_shape, rank=1)
result, history = tt_sum_gmres(map_sum, B_pr, max_rank=100, maxiter=20)
# %%
history["residual_norm"]
# %%

map_sum(result).error(B, relative=True)
# %%
%prun tt_sum_gmres(map_sum, B, max_rank=100, maxiter=10)
# %%
np.mean(A_precond-np.diag(np.diag(A_precond))==0)