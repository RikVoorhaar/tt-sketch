# %%
L = [54,32,10,2]
list(enumerate(L))[::-1]


# %%
from tt_sketch.sketch_container import SketchContainer

from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.tensor import Tensor, DenseTensor, TensorTrain

shape = (5, 6, 7, 8, 10, 5)
rank = 4

X_tt = TensorTrain.random(shape, rank)
X_dense = X_tt.dense()
# %%
%%prun 
for _ in range(200):
    sk_tt = stream_sketch(X_dense, left_rank=rank, right_rank=rank)

# %%

stt = SketchContainer(sk_tt.Psi_cores, sk_tt.Omega_mats)
stt.shape, stt.left_rank, stt.right_rank

# %%


# %%