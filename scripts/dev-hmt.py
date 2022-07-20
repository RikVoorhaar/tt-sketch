# %%
import numpy as np
from tt_sketch.drm import TensorTrainDRM, DenseGaussianDRM
from tt_sketch.sketch_dispatch import general_sketch, SketchMethod
from tt_sketch.sketch import hmt_sketch, orthogonal_sketch
from tt_sketch.tensor import TensorTrain
from tt_sketch.utils import process_tt_rank

shape = (8, 12, 15, 10)
big_rank = process_tt_rank(15, shape, trim=True)
tensor = (
    TensorTrain.random(shape, big_rank)
    + 1e-16 * TensorTrain.random(shape, 1)
    + 1e-20 * TensorTrain.random(shape, 1)
)
small_rank_left = process_tt_rank(18, shape, trim=True)
small_rank_right = tuple(s + 2 for s in small_rank_left)

drm_left = DenseGaussianDRM(small_rank_left, shape, transpose=False)
drm_right_small = DenseGaussianDRM(small_rank_left, shape, transpose=True)
drm_right = DenseGaussianDRM(small_rank_right, shape, transpose=True)

drm_left,drm_right_small.T
# %%
method = SketchMethod.hmt
sketch = general_sketch(tensor, None, drm_right_small, method)
tts = TensorTrain(sketch.Psi_cores)
error_hmt = tts.error(tensor, relative=True)
print(f"Error HMT: {error_hmt:.3e}")

# %%

drm_right_small.sketching_mats[0].dtype
A = np.random.normal(size=(10,10))
np.linalg.qr(A)[0].dtype
# %%
method = SketchMethod.orthogonal
sketch = general_sketch(tensor, drm_left, drm_right, method)
tts = TensorTrain(sketch.Psi_cores)
error_orth = tts.error(tensor, relative=True)
print(f"Error Orth: {error_orth:.3e}")

print(f"Error HMT/Orth: {(error_hmt+1e-10)/(error_orth+1e-10):.2f}")
# %%
drm_right_small.T.rank, drm_left.rank
# %%
error_hmt = hmt_sketch(tensor, small_rank_left).error(tensor, relative=True)
error_orth = orthogonal_sketch(tensor, small_rank_left, small_rank_right).error(
    tensor, relative=True
)

print(f"Error HMT: {error_hmt:.3e}")
print(f"Error Orth: {error_orth:.3e}")
print(f"Error HMT/Orth: {(error_hmt+1e-20)/(error_orth+1e-20):.2f}")

# %%