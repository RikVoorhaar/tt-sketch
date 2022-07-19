# from tt_sketch.tt_gmres import (
#     tt_weighted_sum_sketched,
#     tt_weighted_sum_exact,
#     tt_gmres,
# )
import pytest
from tt_sketch.tt_gmres import MPO
from tt_sketch.tensor import TensorTrain
import numpy as np


def test_mpo_contract():
    shape = (4, 5, 6)

    mpo = MPO.random(3, shape, shape)

    mpo_dense = mpo.to_numpy()

    tt = TensorTrain.random(shape, 2)

    mpo_dense_t = mpo_dense.transpose((1, 0, 3, 2, 5, 4))
    assert np.linalg.norm(mpo_dense - mpo_dense_t) < 1e-12  # assert symmetry

    tt_dense = tt.to_numpy()

    tt_mpo1 = np.einsum("ijk,iajbkc", tt_dense, mpo_dense)
    tt_mpo2 = mpo(tt).to_numpy()

    assert np.linalg.norm(tt_mpo1 - tt_mpo2) < 1e-12


# @pytest.mark.parametrize("method", ["sketched", "exact"])
# def test_tt_weighted_sum(method):
#     shape = (4, 5, 6)
#     n_elems = 10
#     x0 = TensorTrain.random(shape, 3)
#     tt_list = [TensorTrain.random(shape, 3) for _ in range(n_elems)]
#     coeffs = np.random.normal(size=n_elems)

#     dense = x0.to_numpy()
#     for tt, coeff in zip(tt_list, coeffs):
#         dense += coeff * tt.to_numpy()

#     if method == "exact":
#         tt_sum = tt_weighted_sum_exact(x0, coeffs, tt_list, 1e-6, 6)
#     if method == "sketched":
#         tt_sum = tt_weighted_sum_sketched(x0, coeffs, tt_list, 1e-6, 6)
#     assert np.linalg.norm(tt_sum.to_numpy() - dense) < 1e-8


# def test_tt_gmres():
#     shape = (3, 3, 3)
#     mpo = MPO.random(4, shape, shape)
#     tt_true = TensorTrain.random(shape, 2)
#     tt = mpo(tt_true)

#     rank = 6
#     tt_approx, _ = tt_gmres(mpo, tt, rank, maxiter=100, tolerance=1e-8)
#     assert mpo(tt_approx).mse_error(tt) / tt.norm() < 1e-6
