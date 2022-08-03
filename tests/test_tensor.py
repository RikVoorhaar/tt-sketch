import enum
import pytest
from sklearn.metrics import auc
from tt_sketch.tensor import (
    CPTensor,
    DenseTensor,
    SparseTensor,
    TensorTrain,
    TuckerTensor,
)
import numpy as np


def test_tt_round():
    shape = (5, 5, 5, 5)
    ndim = len(shape)
    tt = TensorTrain.random(shape, 2)
    new_cores = list(tt.cores)
    new_rank = 5
    for i in range(ndim - 1):
        r = tt.rank[i]
        Q = np.random.normal(size=(new_rank, r))
        Q, _ = np.linalg.qr(Q)
        new_cores[i] = np.einsum("ijk,kl->ijl", new_cores[i], Q.T)
        new_cores[i + 1] = np.einsum("ij,jkl->ikl", Q, new_cores[i + 1])
    tt2 = TensorTrain(new_cores)

    assert tt.error(tt2) < 1e-8

    tt3 = tt2.round(eps=1e-12)
    assert tt3.rank == tt.rank
    assert tt2.rank != tt.rank  # check for side effects
    assert tt.error(tt3) < 1e-8

    tt4 = tt2.round(max_rank=2)
    assert tt4.rank == tt.rank
    assert tt.error(tt4) < 1e-8


def test_tt_add():
    shape = (5, 5, 5, 5)
    rank = 2
    tt1 = TensorTrain.random(shape, rank)
    tt2 = TensorTrain.random(shape, rank)
    tt3 = tt1.add(tt2)
    assert tt3.rank == (4, 4, 4)
    assert (
        np.linalg.norm(tt3.to_numpy() - tt1.to_numpy() - tt2.to_numpy()) < 1e-8
    )
    tt0 = TensorTrain.zero(shape, rank)

    tt12 = tt1 + tt2
    tt120 = tt12 + tt0
    tt123 = tt12 + tt3
    tt312 = tt3 + tt12
    tt1212 = tt12 + tt12
    assert (
        np.linalg.norm(tt12.to_numpy() - tt1.to_numpy() - tt2.to_numpy()) < 1e-8
    )
    assert np.linalg.norm(tt12.to_numpy() - tt120.to_numpy()) < 1e-8
    assert (
        np.linalg.norm(
            tt1212.to_numpy() - 2 * tt1.to_numpy() - 2 * tt2.to_numpy()
        )
        < 1e-8
    )

    assert (
        np.linalg.norm(
            tt123.to_numpy() - 2 * tt1.to_numpy() - 2 * tt2.to_numpy()
        )
        < 1e-8
    )
    assert (
        np.linalg.norm(
            tt312.to_numpy() - 2 * tt1.to_numpy() - 2 * tt2.to_numpy()
        )
        < 1e-8
    )


def test_tt_dot():
    shape = (5, 5, 5, 5)
    rank = 2

    tt1 = TensorTrain.random(shape, rank)
    tt2 = TensorTrain.random(shape, rank)

    prod1 = tt1.dot(tt2)
    prod2 = tt1.to_numpy().reshape(-1).dot(tt2.to_numpy().reshape(-1))
    assert abs(prod1 - prod2) < 1e-8

    tt3 = tt1.orthogonalize()
    assert abs(np.linalg.norm(tt3[-1]) - tt3.norm()) < 1e-8


def make_cp(shape):
    return CPTensor.random(shape, 10)


def make_tt(shape):
    return TensorTrain.random(shape, 5)


def make_tucker(shape):
    return TuckerTensor.random(shape, 5)


def make_sparse(shape):
    return SparseTensor.random(shape, 5)


def make_dense(shape):
    return DenseTensor.random(shape)


class TensorType(enum.Enum):
    CP = enum.auto()
    TT = enum.auto()
    TUCKER = enum.auto()
    SPARSE = enum.auto()
    DENSE = enum.auto()


make_tensor_dispatch = {
    TensorType.CP: make_cp,
    TensorType.TT: make_tt,
    TensorType.TUCKER: make_tucker,
    TensorType.SPARSE: make_sparse,
    TensorType.DENSE: make_dense,
}


@pytest.mark.parametrize("tensor_type1", make_tensor_dispatch.keys())
@pytest.mark.parametrize("tensor_type2", make_tensor_dispatch.keys())
def test_dot(tensor_type1, tensor_type2):
    shape = (5, 6, 7, 4)
    tensor1 = make_tensor_dispatch[tensor_type1](shape)
    tensor2 = make_tensor_dispatch[tensor_type2](shape)
    dot_fast = tensor1 @ tensor2
    dot_slow = (
        tensor1.to_numpy().reshape(-1).dot(tensor2.to_numpy().reshape(-1))
    )
    assert abs(dot_fast - dot_slow) < 1e-8

    assert tensor1.to_numpy().shape == shape
    assert tensor2.to_numpy().shape == shape

    norm_fast = tensor1.norm()
    assert abs(norm_fast - np.linalg.norm(tensor1.to_numpy())) < 1e-8

    error_fast = tensor1.error(tensor2, fast=True)
    error_slow = np.linalg.norm(tensor1.to_numpy() - tensor2.to_numpy())
    assert abs(error_fast - error_slow) < 1e-8

    error_relative = tensor1.error(tensor2, relative=True)
    error_slow_relative = error_slow / tensor2.norm()
    assert abs(error_relative - error_slow_relative) < 1e-8

    error_inf = tensor1.error(0 * tensor2, relative=True)
    assert error_inf == np.inf

    error_rmse = tensor1.error(tensor2, rmse=True)
    error_rmse_slow = np.sqrt(
        np.mean((tensor1.to_numpy() - tensor2.to_numpy()) ** 2)
    )
    assert abs(error_rmse - error_rmse_slow) < 1e-8


def test_error_tt():
    shape = (5, 5, 5, 5)
    rank = 2
    tt1 = TensorTrain.random(shape, rank)
    tt2 = TensorTrain.random(shape, rank)
    error = tt1.error(tt2)
    assert abs(error - np.linalg.norm(tt1.to_numpy() - tt2.to_numpy())) < 1e-8

    tt3 = tt1 + 1e-10 * tt2
    error = tt1.error(tt3)
    assert abs(error - np.linalg.norm(tt1.to_numpy() - tt3.to_numpy())) < 1e-8


@pytest.mark.parametrize("tensor_type1", make_tensor_dispatch.keys())
@pytest.mark.parametrize("tensor_type2", make_tensor_dispatch.keys())
def test_arithmetic(tensor_type1, tensor_type2):
    shape = (5, 6, 7, 4)
    tensor1 = make_tensor_dispatch[tensor_type1](shape)
    tensor2 = make_tensor_dispatch[tensor_type2](shape)

    tensor3 = 2.3 * tensor1 - (1.5 * tensor2)
    assert (
        abs(
            np.linalg.norm(
                tensor3.to_numpy()
                - (2.3 * tensor1.to_numpy() - 1.5 * tensor2.to_numpy())
            )
        )
        < 1e-8
    )

    tensor4 = -tensor1 / 2 + (-1.5) * tensor2
    assert (
        abs(
            np.linalg.norm(
                tensor4.to_numpy()
                - (-tensor1.to_numpy() / 2 + (-1.5) * tensor2.to_numpy())
            )
        )
        < 1e-8
    )

    tensor5 = tensor3 + tensor4
    assert (
        abs(
            np.linalg.norm(
                tensor5.to_numpy() - (tensor3.to_numpy() + tensor4.to_numpy())
            )
        )
        < 1e-8
    )

    tensor6 = 1.4 * tensor3.T - 2 * tensor1.T
    assert (
        abs(
            np.linalg.norm(
                tensor6.to_numpy()
                - (1.4 * tensor3.to_numpy().T - 2 * tensor1.to_numpy().T)
            )
        )
        < 1e-8
    )

    tensor6 += tensor5.T
    assert (
        abs(
            np.linalg.norm(
                tensor6.to_numpy()
                - (1.4 * tensor3.to_numpy().T - 2 * tensor1.to_numpy().T+tensor5.to_numpy().T)
            )
        )
        < 1e-8
    )