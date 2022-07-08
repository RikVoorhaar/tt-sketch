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
from time import perf_counter
from tqdm import tqdm
from tt_sketch.sketch import orthogonal_sketch, stream_sketch
from tt_sketch.tensor import Tensor, TensorSum, TensorTrain
from tt_sketch.tt_gmres import (
    TTLinearMap,
    tt_weighted_sum,
    round_tt_sum,
    TTLinearMapInverse,
    TTLinearMapSum,
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
        new_cores[0] = np.einsum("ijk,jl->ilk", new_cores[0], self.A)
        if self.mode != 0:
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
            coeffs = np.linspace(0, 10, num_coeffs, dtype=np.float64)
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


# TODO: keep precond separate, needed to compute correct error
# TODO: Support two-stage rounding at end
# TODO: Make experiment dataframes for plotting
# TODO: Support stream-sketch and orth-sketch separately
# TODO: Better verbosity
# TODO: Make version where all computations are done in a sketch-range
def tt_sum_gmres(
    A: TTLinearMapSum,
    b: TensorTrain,
    max_rank: TTRank,
    x0: Optional[TensorTrain] = None,
    tolerance: float = 1e-6,
    maxiter: int = 100,
    symmetric: bool = False,
    exact: bool = False,
    save_basis: bool = False,
    verbose: bool = False,
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
    initial_time = perf_counter()
    residual = b - A(x0)
    residual = round_tt_sum(residual, max_rank=max_rank, exact=exact)
    residual_norm = residual.norm()
    beta = residual_norm
    nu_list: List[TensorTrain] = [residual / beta]
    H_matrix = np.zeros((maxiter + 1, maxiter))

    history: Dict[str, List] = defaultdict(list)
    history["w_norm"].append(nu_list[-1].norm())
    history["rank"].append(residual.rank)
    history["residual_norm"].append(residual_norm / b_norm)
    history["step_time"].append(perf_counter() - initial_time)

    for j in range(maxiter):
        current_time = perf_counter()
        delta = tolerance / (residual_norm / beta)
        if verbose:
            print(j, history["residual_norm"][-1], history["rank"][-1])
        w = A(nu_list[-1])
        w = round_tt_sum(w, eps=delta, max_rank=max_rank, exact=exact)

        min_j = max(0, j - 2) if symmetric else 0
        for i in range(min_j, j + 1):
            H_matrix[i, j] = w.dot(nu_list[i])

        w = tt_weighted_sum(
            w,
            -H_matrix[min_j : j + 1, j],
            nu_list[min_j : j + 1],
            tolerance,
            max_rank,
            exact=exact,
            oversample=20,
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
        history["step_time"].append(perf_counter() - current_time)

        if residual_norm / b_norm < tolerance:
            break

    y = y[: j + 1]
    nu_list = nu_list[: j + 1]
    current_time = perf_counter()
    x = tt_weighted_sum(
        x0,
        y,
        nu_list,
        tolerance,
        max_rank,
        exact=exact,
        oversample=10,
    )
    history["final_round_time"] = perf_counter() - current_time
    history["total_time"] = perf_counter() - initial_time
    if save_basis:
        history["H_matrix"] = H_matrix
        history["nu_list"] = nu_list
        history["y"] = y
    return x, history


# %%
map_sum, B, B_pr, precond_map = prepare_cookie_problem(10, 4)

x0 = TensorTrain.zero(shape=map_sum.in_shape, rank=10)
max_rank = 20
result, history = tt_sum_gmres(
    A=map_sum,
    b=B_pr,
    x0=x0,
    max_rank=max_rank,
    maxiter=50,
    tolerance=1e-20,
    symmetric=True,
    exact=False,
    save_basis=True,
    verbose=False
)
print(map_sum(result).error(B_pr, relative=True))
# %%
history["total_time"], history["total_time"] - history["final_round_time"]
# %%
sketch_rank = 20
tt_sketched = stream_sketch(x0, left_rank=sketch_rank, right_rank=sketch_rank+100)
errors = []
error = map_sum(tt_sketched.to_tt()).error(B_pr, relative=True)
errors.append(error)
for y,nu in zip(history["y"], history["nu_list"]):
    tt_sketched += y*nu
    error = map_sum(tt_sketched.to_tt()).error(B_pr, relative=True)
    errors.append(error)

plt.figure(figsize=(10, 5))
svdvals = tt_sketched.to_tt().svdvals()
for mu, S in enumerate(svdvals):
    plt.plot(S / S[0], label=f"mode={mu}")
plt.yscale("log")
plt.legend()
# %%
plt.plot(errors)
plt.yscale('log')
# %%
# %%
rounded = tt_sketched.to_tt().round(eps=1e-5)
error = map_sum(tt_sketched.to_tt()).error(B_pr, relative=True)
error
# %%
rounded
# %%
