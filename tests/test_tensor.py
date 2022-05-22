import pytest
from tt_sketch.tensor import TensorTrain
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

    assert tt.mse_error(tt2) < 1e-8

    tt3 = tt2.round(eps=1e-12)
    assert tt3.rank == tt.rank
    assert tt2.rank != tt.rank  # check for side effects
    assert tt.mse_error(tt3) < 1e-8

    tt4 = tt2.round(max_rank=2)
    assert tt4.rank == tt.rank
    assert tt.mse_error(tt4) < 1e-8


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
    tt120 = tt12+tt0
    tt123 = tt12 + tt3
    tt312 = tt3 + tt12
    tt1212 = tt12 + tt12
    assert (
        np.linalg.norm(tt12.to_numpy() - tt1.to_numpy() - tt2.to_numpy()) < 1e-8
    )
    assert (
        np.linalg.norm(tt12.to_numpy() - tt120.to_numpy()) < 1e-8
    )
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
