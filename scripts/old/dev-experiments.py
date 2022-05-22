# %%

import sys

import numpy as np
import pytest

sys.path.append("..")

from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.drm import (
    ALL_DRM,
    SparseGaussianDRM,
    DenseGaussianDRM,
    TensorTrainDRM,
    SparseSignDRM,
)
from tt_sketch.sketching_methods import CansketchSparse, CansketchTT
from tt_sketch.tensor import SparseTensor, TensorTrain

sparse_drm_list = [
    drm_type for drm_type in ALL_DRM if issubclass(drm_type, CansketchSparse)
]
seed = 180
n_dims = 6
rank = 3
X_shape = tuple(range(7, 7 + n_dims))
X_tt = TensorTrain.random(X_shape, rank)
X = X_tt.to_numpy()

left_rank = tuple(range(rank, rank + n_dims - 1))
right_rank = tuple(range(rank + 1, rank + n_dims))
print(left_rank, right_rank)

X_sparse = X_tt.dense().to_sparse()

failure_cases = []
left_drm = SparseSignDRM
right_drm = SparseSignDRM
print(f"{left_drm=}, {right_drm=}")
# %%
%%prun
tt_sketched = stream_sketch(
    X_sparse,
    left_rank,
    right_rank,
    seed=seed,
    left_sketch_type=left_drm,
    right_sketch_type=right_drm,
)

error = np.linalg.norm(TensorTrain(tt_sketched.C_cores).to_numpy() - X)
error
# %%
