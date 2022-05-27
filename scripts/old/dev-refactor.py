# %%
from tt_sketch import utils
from tt_sketch.sketch import stream_sketch
from tt_sketch.tensor import DenseTensor, TensorSum, TensorTrain
from tt_sketch.drm import TensorTrainDRM, SparseGaussianDRM
import numpy as np

seed = 180
n_dims = 3
rank = 5
X_shape = tuple(range(9, 9 + n_dims))
X_tt = TensorTrain.random(X_shape, rank)
X = X_tt.to_numpy()

left_rank = tuple(range(rank, rank + n_dims - 1))
right_rank = tuple(range(rank + 1, rank + n_dims))
left_rank = 100
right_rank = 80

X_sparse = X_tt.dense().to_sparse()

stt = stream_sketch(
    X_sparse,
    left_rank,
    right_rank,
    seed=seed,
    left_drm_type=SparseGaussianDRM,
    right_drm_type=TensorTrainDRM,
)

left_tt = TensorTrain(stt.C_cores(direction="left"))
right_tt = TensorTrain(stt.C_cores(direction="right"))

assert np.allclose(left_tt.to_numpy(), right_tt.to_numpy())
assert left_tt.rank == stt.right_rank
assert right_tt.rank == stt.left_rank
stt
# %%
import sys

sys.path.append("../tests")
from test_sketching_matrix import test_tensor_sum_parallel
from test_tt_gmres import test_tt_gmres

test_tt_gmres()
# %%
left_rank = (2, 2, 2)
right_rank = (3, 4, 5)
np.all(np.array(left_rank) < np.array(right_rank))
# %%
shape = (5, 5, 5, 5)
rank = 2
tt1 = TensorTrain.random(shape, rank)
tt2 = TensorTrain.random(shape, rank)
tt3 = tt1.add(tt2)
assert tt3.rank == (4, 4, 4)
assert np.linalg.norm(tt3.to_numpy() - tt1.to_numpy() - tt2.to_numpy()) < 1e-8

tt12 = tt1 + tt2

isinstance(tt12, TensorSum)

# %%
from tt_sketch.sketch import stream_sketch
from tt_sketch.tensor import DenseTensor
from tt_sketch.drm import DenseGaussianDRM




test_sketch_dense(2)
