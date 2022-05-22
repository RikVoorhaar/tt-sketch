# %%
from tt_sketch.tensor import TensorTrain, TensorSum, CPTensor
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.drm import TensorTrainDRM, SparseGaussianDRM
import numpy as np
from copy import copy

rank = 5
dim = 10
n_dim = 5
shape = tuple(dim for _ in range(n_dim))
tt_rank = tuple(rank for _ in range(n_dim-1))
tt1 = TensorTrain.random(shape, rank)
tt_sum = copy(tt1)
for _ in range(2):
    tt_sum += TensorTrain.random(shape, rank)

stt = stream_sketch(tt_sum, 10, 10)
# %%
rank = 10
cp1 = CPTensor.random(shape, rank)
cp_sum = copy(cp1)
for _ in range(1):
    cp_sum += CPTensor.random(shape, rank)

stt = stream_sketch(cp_sum, 10, 10)
# %time stream_sketch(tt1,10,10)
# stt.to_tt().error(tt_sum)

# %%


# %%
def test_tensor_sum_parallel():
    seed = 179
    rank = 4
    n_dims = 4

    X_shape = tuple(range(7, 7 + n_dims))

    X1_tt = TensorTrain.random(X_shape, rank)
    X1 = X1_tt.to_numpy()

    left_rank = tuple(range(rank, rank + n_dims - 1))
    right_rank = tuple(range(rank + 1, rank + n_dims))

    X_sparse = X1_tt.dense().to_sparse()
    X_sparse_sum = X_sparse.split(2)
    assert isinstance(X_sparse_sum, TensorSum)

    left_drm = TensorTrainDRM
    right_drm = TensorTrainDRM

    # Sketching is linear, so doing it as (parallel) sum should give same result
    stt1 = stream_sketch(
        X_sparse_sum, left_rank, right_rank, seed, left_drm, right_drm
    )
    # return
    stt2 = stream_sketch(
        X_sparse, left_rank, right_rank, seed, left_drm, right_drm
    )
    for Y1, Y2 in zip(stt1.Psi_cores, stt2.Psi_cores):
        assert np.allclose(Y1, Y2)
    for Y1, Y2 in zip(stt1.Omega_mats, stt2.Omega_mats):
        assert np.allclose(Y1, Y2)

    X2_tt = TensorTrain.random(X_shape, rank)
    X2 = X1 + X2_tt.to_numpy()

    X1_plus_X2 = X2_tt+X_sparse.split(2)
    print(X1_plus_X2.tensors)
    left_drm = TensorTrainDRM
    right_drm = TensorTrainDRM
    print("so far so good...")
    stt3 = stream_sketch(
        X1_plus_X2, left_rank, right_rank, seed, left_drm, right_drm
    )
    assert stt3.to_tt().mse_error(X2) < 1e-8


test_tensor_sum_parallel()
# %%

# %%
