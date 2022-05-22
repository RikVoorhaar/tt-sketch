# %%
import numpy as np
from tt_sketch.drm import TensorTrainDRM
from tt_sketch.sketching_methods import (
    cp_sketch,
    dense_sketch,
    tensor_train_sketch,
)
from tt_sketch.drm import (
    TensorTrainDRM,
)
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.tt_svd import tt_svd
from tt_sketch.tensor import TensorTrain, DenseTensor, CPTensor
from tt_sketch.utils import matricize
from tree_tt import forest_compression

n_dims = 3
rank = 3
seed = 179

X_shape = tuple(range(7, 7 + n_dims))


X_cp = CPTensor.random(X_shape, rank, seed=seed)
X_dense = X_cp.dense()
X_tt = TensorTrain.from_dense(X_dense)
X = X_cp.to_numpy()

sk = stream_sketch(X_cp, 5, 6)
sk.dense().mse_error(X_dense)

# %%
print("CP")
print("C shapes: ", [C.shape for C in sk.C_cores])
print("Y shapes: ", [C.shape for C in sk.Psi_cores])
print("Z shapes: ", [C.shape for C in sk.Omega_mats])

sk = stream_sketch(X_dense, 5, 6)
print("\nDense")
print("C shapes: ", [C.shape for C in sk.C_cores])
print("Y shapes: ", [C.shape for C in sk.Psi_cores])
print("Z shapes: ", [C.shape for C in sk.Omega_mats])
# %%

drm = TensorTrainDRM(3, X_shape, False, seed=None)
left = drm.sketch_cp(X_cp)
print([L.shape for L in left])

left = drm.sketch_tt(X_tt)
print([L.shape for L in left])

# %%
