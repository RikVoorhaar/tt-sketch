# %%
from operator import ne
from tt_sketch.utils import hilbert_tensor
from tt_sketch.tensor import SparseTensor, DenseTensor, TensorTrain
from tt_sketch.drm import TensorTrainDRM

tensor = hilbert_tensor(4, 10)
shape = tensor.shape
sparse_tensor = DenseTensor(shape, tensor).to_sparse()

rank = 2
cores = TensorTrain.random(shape, rank).cores
drm = TensorTrainDRM(rank, shape, transpose=False, cores=[cores[0]])

sketch_generator = drm.sketch_sparse(sparse_tensor)
next(sketch_generator)
drm.cores.append(cores[1]*0)
next(sketch_generator)


# %%
l = [10,40,50,10]
for i in l:
    print(i)
iter_list = iter(l)
next(iter_list)