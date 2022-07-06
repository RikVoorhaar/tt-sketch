# %%
from __future__ import annotations

from abc import ABC, abstractmethod
import scipy.io


from tt_sketch.tt_gmres import TTLinearMap
from tt_sketch.tensor import TensorSum, TensorTrain, Tensor
from typing import Generic, TypeVar, Union, List, Tuple, Optional

import numpy as np
import numpy.typing as npt


class TTLinearMapSum:
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    linear_maps: List[TTLinearMap]

    def __init__(self, linear_maps: List[TTLinearMap]) -> None:
        self.linear_maps = linear_maps
        if len(linear_maps) == 0:
            raise ValueError("linear_maps cannot be empty")
        in_shape = linear_maps[0].in_shape
        out_shape = linear_maps[0].out_shape
        for linear_map in linear_maps[1:]:
            if linear_map.in_shape != in_shape:
                raise ValueError("in_shape mismatch")
            if linear_map.out_shape != out_shape:
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


cookies_data = scipy.io.loadmat("data/cookies_matrices_3x3.mat")
b = cookies_data["b"].reshape(-1)
A_list = cookies_data["A"][0]


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

    def __call__(self, other: TensorTrain) -> TensorTrain:
        new_cores = other.cores
        new_cores[0] = np.einsum("ijk,jl->ilk", new_cores[0], self.A)
        new_cores[self.mode] = np.einsum(
            "ijk,j->ijk", new_cores[self.mode], self.coeffs
        )
        return TensorTrain(new_cores)


cookie_maps = []

num_coeffs = 100
shape = (A_list[0].shape[0],) + (num_coeffs,) * (len(A_list) - 1)
for mu, A in enumerate(A_list):
    if mu == 0:
        coeffs = np.ones(A.shape[0])
    else:
        coeffs = np.arange(num_coeffs, dtype=np.float64) + 1
    cookie_maps.append(
        CookieMap(
            A.toarray(),
            mu,
            shape,
            coeffs,
        )
    )

map_sum = TTLinearMapSum(cookie_maps)


tt = TensorTrain.random(shape, 10)
map_sum(tt)
# %%

"""
TODO:
- Implement an elegant .dot method
- Make GMRES for TTLinearMapSum; Have rounding as an option.
- Implement preconditioner.
- Test the method on the cookie problem. Tobler's thesis should give reasonable 
  estimates of what we can expect.
"""
