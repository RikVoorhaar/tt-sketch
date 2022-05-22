# %%
from cgi import test
from operator import ne

from matplotlib.pyplot import get
from tt_sketch.utils import hilbert_tensor
from tt_sketch.tensor import (
    SparseTensor,
    DenseTensor,
    TensorTrain,
    SketchedTensorTrain,
    CPTensor,
)
from tt_sketch.drm import TensorTrainDRM, DenseGaussianDRM
from tt_sketch.sketching_methods.tensor_train_sketch import (
    tensor_train_sketch,
)
from tt_sketch.sketch_dispatch import general_sketch, get_sketch_method
from tt_sketch.sketching_methods.sparse_sketch import sparse_sketch
from tt_sketch.sketching_methods.cp_sketch import cp_sketch

tensor = hilbert_tensor(4, 10)
shape = tensor.shape
sparse_tensor = DenseTensor(shape, tensor).to_sparse()

rank = 2
cores = TensorTrain.random(shape, rank).cores
drm = TensorTrainDRM(rank, shape, transpose=False, cores=[cores[0]])

sketch_generator = drm.sketch_sparse(sparse_tensor)
next(sketch_generator)
drm.cores.append(cores[1] * 0)
next(sketch_generator)


# %%

shape = (10, 3, 5)
rank = 2
tt_rank = (2, 3)
tt = TensorTrain.random(shape, tt_rank)
left_rank = tt_rank
right_rank = tuple(r + 1 for r in tt_rank)
left_drm = TensorTrainDRM(left_rank, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank, shape, transpose=True)
sketch_stream = general_sketch(tt, left_drm, right_drm)
sketch_orthog = general_sketch(tt, left_drm, right_drm, orthogonalize=True)
tt_orthog = TensorTrain(sketch_orthog.Psi_cores)
tt_stream = SketchedTensorTrain(sketch_stream).to_tt()
tt_orthog.mse_error(tt), tt_stream.mse_error(tt)

# %%
sparse_tensor = DenseTensor(tt.shape, tt.to_numpy()).to_sparse()
left_drm = TensorTrainDRM(left_rank, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank, shape, transpose=True)
sketch_stream = general_sketch(sparse_tensor, left_drm, right_drm)
sketch_orthog = general_sketch(
    sparse_tensor, left_drm, right_drm, orthogonalize=True
)
tt_orthog = TensorTrain(sketch_orthog.Psi_cores)
tt_stream = SketchedTensorTrain(sketch_stream).to_tt()
tt_orthog.mse_error(tt), tt_stream.mse_error(tt)
# %%
left_rank = rank
right_rank = rank
cp_tensor = CPTensor.random(shape, rank)
left_drm = TensorTrainDRM(left_rank, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank, shape, transpose=True)
sketch_stream = general_sketch(cp_tensor, left_drm, right_drm)
sketch_orthog = general_sketch(
    cp_tensor, left_drm, right_drm, orthogonalize=True
)
tt_orthog = TensorTrain(sketch_orthog.Psi_cores)
tt_stream = SketchedTensorTrain(sketch_stream).to_tt()
tt_orthog.mse_error(cp_tensor), tt_stream.mse_error(cp_tensor)

# %%
tensor_sum = cp_tensor + sparse_tensor
left_drm = TensorTrainDRM(left_rank + 3, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank + 10, shape, transpose=True)
sketch_stream = general_sketch(tensor_sum, left_drm, right_drm)
sketch_orthog = general_sketch(
    tensor_sum, left_drm, right_drm, orthogonalize=True
)
tt_orthog = TensorTrain(sketch_orthog.Psi_cores)
tt_stream = SketchedTensorTrain(sketch_stream).to_tt()
tt_orthog.mse_error(tensor_sum), tt_stream.mse_error(tensor_sum)
# %%
tensor_numpy = hilbert_tensor(4, 10)
shape = tensor_numpy.shape
tensor_dense = DenseTensor(shape, tensor_numpy)
left_drm = TensorTrainDRM(left_rank + 3, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank + 10, shape, transpose=True)
print([M.shape for M in left_drm.sketch_dense(tensor_dense)])
print([M.shape for M in right_drm.sketch_dense(tensor_dense)])

left_drm = DenseGaussianDRM(left_rank + 3, shape, transpose=False)
right_drm = DenseGaussianDRM(right_rank + 10, shape, transpose=True)
print([M.shape for M in left_drm.sketch_dense(tensor_dense)])
print([M.shape for M in right_drm.sketch_dense(tensor_dense)])

# %%
tensor_numpy = hilbert_tensor(4, 5)
shape = tensor_numpy.shape
tensor_dense = DenseTensor(shape, tensor_numpy)
left_drm = TensorTrainDRM(left_rank + 3, shape, transpose=False)
right_drm = TensorTrainDRM(right_rank + 10, shape, transpose=True)
print(left_drm.rank)
print(right_drm.rank)
sketch_stream = general_sketch(tensor_dense, left_drm, right_drm)
try:
    sketch_orthog = general_sketch(
        tensor_dense, left_drm, right_drm, orthogonalize=True
    )
except StopIteration:
    raise ValueError("Sketch failed")
tt_orthog = TensorTrain(sketch_orthog.Psi_cores)
tt_stream = SketchedTensorTrain(sketch_stream).to_tt()
tt_orthog.mse_error(tensor_dense), tt_stream.mse_error(tensor_dense)


# %%

import sys

sys.path.append("../test")
from tests.test_sketching_matrix import (
    test_exact_recovery_sparse,
    test_sketch_dense,
    test_tensor_sum_parallel,
)

test_exact_recovery_sparse(2, 3, "TensorTrainDRM|TensorTrainDRM", True)
# test_tensor_sum_parallel()

# %%
from tt_sketch.utils import matricize
from tt_sketch.drm import DenseGaussianDRM
import numpy as np

X = np.random.normal(size=(10, 5, 8, 9))
X_dense = DenseTensor(X.shape, X)
matricize(X, range(2), mat_shape=True).shape
left_drm = DenseGaussianDRM(2, X.shape, transpose=False)
right_drm = DenseGaussianDRM(3, X.shape, transpose=True)
[m.shape for m in right_drm.sketch_dense(X)]

# %%
