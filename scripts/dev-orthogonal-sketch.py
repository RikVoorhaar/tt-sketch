# %%
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
from tt_sketch.drm import TensorTrainDRM
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

left_rank = rank
right_rank = rank + 1
tt = TensorTrain.random(shape, rank)
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
list(get_sketch_method(tensor_sum, left_drm)(tensor_sum))
list(get_sketch_method(sparse_tensor, left_drm)(sparse_tensor))