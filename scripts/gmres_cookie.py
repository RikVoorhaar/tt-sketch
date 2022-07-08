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
from tt_sketch.tt_gmres import (
    TTLinearMap,
    tt_weighted_sum,
    round_tt_sum,
    TTLinearMapInverse,
    TTLinearMapSum,
    tt_weighted_sum_exact,
    tt_weighted_sum_sketched,
)
from tt_sketch.utils import (
    ArrayList,
    TTRank,
    dematricize,
    matricize,
    process_tt_rank,
    trim_ranks,
)


COOKIES_2x2_FILE = "data/cookies_matrices_2x2.mat"
COOKIES_3x3_FILE = "data/cookies_matrices_3x3.mat"


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
            "ijk,jl->ilk", new_cores[0], self.A
        )
        new_cores[self.mode] = np.einsum(
            "ijk,j->ijk", new_cores[self.mode], self.coeffs
        )
        tt = TensorTrain(new_cores)
        if self.preconditioner is not None:
            tt = self.preconditioner(tt)
        return tt

    __call__ = cookie_call


def prepare_cookie_problem(num_coeffs, num_cookies):
    if num_cookies == 4:
        cookies_file = COOKIES_2x2_FILE
    elif num_cookies == 9:
        cookies_file = COOKIES_3x3_FILE
    else:
        raise ValueError("num_cookies must be 4 or 9")
    cookies_data = scipy.io.loadmat(cookies_file)
    b = cookies_data["b"].reshape(-1)
    A_list = cookies_data["A"][0]

    shape = (A_list[0].shape[0],) + (num_coeffs,) * (len(A_list) - 1)

    A_precond_list = []
    coeffs_list = []
    for mu, A in enumerate(A_list):
        A = A.toarray()
        if mu == 0:
            coeffs = np.ones(A.shape[0])
        else:
            coeffs = -(np.linspace(0, 10, num_coeffs, dtype=np.float64))
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

    B_cores = [b.reshape(1, -1, 1)]
    for n in shape[1:]:
        B_cores.append(np.ones((1, n, 1)))
    B = TensorTrain(B_cores)
    B_pr = precond_map(B)

    return map_sum, B, B_pr, precond_map


def tt_sum_gmres(
    A: TTLinearMapSum,
    b: TensorTrain,
    max_rank: TTRank,
    x0: Optional[TensorTrain] = None,
    tolerance: float = 1e-6,
    maxiter: int = 100,
    symmetric: bool = False,
    exact: bool = False,
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
        x0 = TensorTrain.zero(shape=A.in_shape, rank=1)

    b_norm = b.norm()
    residual = b - A(x0)
    residual = round_tt_sum(residual, max_rank=max_rank)
    residual_norm = residual.norm()
    beta = residual_norm
    nu_list: List[TensorTrain] = [residual / beta]
    H_matrix = np.zeros((maxiter + 1, maxiter))

    history: Dict[str, List] = defaultdict(list)
    history["w_norm"].append(nu_list[-1].norm())
    history["rank"].append(residual.rank)
    history["residual_norm"].append(residual_norm / b_norm)

    tt_weighted_sum_func = (
        tt_weighted_sum_exact if exact else tt_weighted_sum_sketched
    )

    for j in range(maxiter):
        delta = tolerance / (residual_norm / beta)
        print(j, history["residual_norm"][-1], history["rank"][-1])
        w = A(nu_list[-1])
        w = round_tt_sum(w, eps=delta, max_rank=max_rank, exact=exact)

        min_j = max(0, j - 2) if symmetric else 0
        for i in range(min_j, j + 1):
            H_matrix[i, j] = w.dot(nu_list[i])

        w = tt_weighted_sum_func(
            w,
            -H_matrix[min_j : j + 1, j],
            nu_list[min_j : j + 1],
            tolerance,
            max_rank,
            # round=True,
        )
        H_matrix[j + 1, j] = w.norm()
        nu_list.append(w / H_matrix[j + 1, j])

        H_red = H_matrix[: j + 2, : j + 1]
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, (residual_norm,), _, _ = np.linalg.lstsq(H_red, e1, rcond=None)
        history["residual_norm"].append(np.sqrt(residual_norm) / b_norm)
        history["rank"].append(w.rank)
        history["w_norm"].append(H_matrix[j + 1, j])
        history["delta"].append(delta)

        if residual_norm / b_norm < tolerance:
            break

    x = tt_weighted_sum_sketched(
        x0, y[: j + 1], nu_list[: j + 1], tolerance, max_rank, round=True
    )
    history["H_matrix"] = H_matrix
    return x, history


# %%
map_sum, B, B_pr, precond_map = prepare_cookie_problem(10, 4)

x0 = TensorTrain.zero(shape=map_sum.in_shape, rank=10)
max_rank = (1000, 100, 100, 10)
max_rank = 50
result, history = tt_sum_gmres(
    A=map_sum,
    b=B_pr,
    x0=x0,
    max_rank=max_rank,
    maxiter=50,
    symmetric=True,
    exact=False,
)
print(map_sum(result).error(B_pr, relative=True))
# %%
result, history = tt_sum_gmres(
    A=map_sum, b=B_pr, x0=result, max_rank=max_rank, maxiter=50, symmetric=True
)
print(map_sum(result).error(B_pr, relative=True))
# %%
B_pr.norm()
map_sum(x0).norm()
# %%
plt.plot(history["residual_norm"])
plt.yscale("log")
# %%
plt.matshow(history["H_matrix"])
# %%

# %%
# %prun tt_sum_gmres(map_sum, B, max_rank=100, maxiter=10)
# %%
# np.mean(A_precond - np.diag(np.diag(A_precond)) == 0)
H = history["H_matrix"]
H_aug = np.zeros((H.shape[0], H.shape[1] + 1))
H_aug[:, :-1] = H
e = np.zeros(H.shape[0])
e[0] = 1
scipy.linalg.solve(H_aug, e)
