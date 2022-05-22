# %%
# Theory seems to predict that we get better behavior when using _negative_
# oversampling for the recursive big sketch algorithm. Can we confirm this
# experimentally?


import numpy as np
from tt_sketch.recursive_big_sketch import recursive_big_sketch

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tt_sketch.drm import (
    ALL_DRM,
    DenseGaussianDRM,
    SparseGaussianDRM,
    SparseSignDRM,
    TensorTrainDRM,
)
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.sketching_methods import CansketchDense, CansketchSparse
from tt_sketch.tensor import DenseTensor, dense_to_sparse
from tt_sketch.tt_svd import tt_svd

from tt_sketch.tensor import Tensor, TensorTrain

from experiment_base import (
    Experiment,
    experiment_recursive_sketch,
    experiment_tensor_sketch,
    experiment_tt_svd,
)


def hilbert_tensor(n_dims: int, size: int) -> DenseTensor:
    grid = np.meshgrid(*([np.arange(size)] * n_dims))
    hilbert = 1 / (np.sum(np.array(grid), axis=0) + 1)
    return DenseTensor((size,) * n_dims, hilbert)


size = 8
n_dims = 6
hilbert = hilbert_tensor(n_dims, size).to_sparse()

r = 3
l = 2
left_rank = (r,) * (n_dims - 1)
right_rank = (r + l,) * (n_dims - 1)

cores = recursive_big_sketch(hilbert.to_numpy(), left_rank, right_rank)
tt = TensorTrain(cores)
error = tt.mse_error(hilbert)
print(error)

cores = recursive_big_sketch(hilbert.to_numpy(), right_rank, left_rank)
tt = TensorTrain(cores)
error = tt.mse_error(hilbert)
print(error)
