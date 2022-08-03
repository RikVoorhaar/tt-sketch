"""Solve the cookie problem using TT-GMRES. 

The files ``cookies_matrices_2x2.mat`` and ``cookies_matrices_3x3.mat`` can be obtained from the 'examples' folder of the ``htucker`` package.
https://www.epfl.ch/labs/anchp/index-html/software/htucker/
"""
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
from tt_sketch.sketch import stream_sketch
from tt_sketch.tensor import TensorSum, TensorTrain
from tt_sketch.tt_gmres import (
    TTLinearMap,
    TTPrecond,
    TTLinearMapSum,
)
from tt_sketch.utils import (
    TTRank,
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
    ) -> None:
        self.A = A
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
    precond_map = TTPrecond(A_precond, shape, mode=0)

    cookie_maps = []

    for mu, (A, coeffs) in enumerate(zip(A_list, coeffs_list)):
        cookie_maps.append(
            CookieMap(
                A.toarray(),
                mu,
                shape,
                coeffs,
            )
        )

    map_sum = TTLinearMapSum(cookie_maps)

    B_cores = [b.reshape(1, -1, 1)]
    for n in shape[1:]:
        B_cores.append(np.ones((1, n, 1)))
    B = TensorTrain(B_cores)

    return map_sum, B, precond_map

