# %%
import numpy as np
from tt_sketch.drm import TensorTrainDRM, DenseGaussianDRM
from tt_sketch.sketch_dispatch import general_sketch, SketchMethod
from tt_sketch.tensor import TensorTrain
from tt_sketch.utils import process_tt_rank

shape = (10, 10, 10, 10)
big_rank = process_tt_rank(15, shape, trim=True)
tensor = (
    TensorTrain.random(shape, big_rank)
    + 1e-4 * TensorTrain.random(shape, big_rank)
    + 1e-6 * TensorTrain.random(shape, big_rank)
)
small_rank_left = process_tt_rank(15, shape, trim=True)
small_rank_right = tuple(s + 2 for s in small_rank_left)

drm_left = DenseGaussianDRM(small_rank_left, shape, transpose=False)
drm_right = DenseGaussianDRM(small_rank_right, shape, transpose=True)

method = SketchMethod.hmt
sketch = general_sketch(tensor, drm_left, drm_left.T, method)
tts = TensorTrain(sketch.Psi_cores)
error_hmt = tts.error(tensor, relative=True)
print(f"Error HMT: {error_hmt:.3e}")

method = SketchMethod.orthogonal
sketch = general_sketch(tensor, drm_left, drm_right, method)
tts = TensorTrain(sketch.Psi_cores)
error_orth = tts.error(tensor, relative=True)
print(f"Error Orth: {error_orth:.3e}")

print(f"Error HMT/Orth: {(error_hmt+1e-10)/(error_orth+1e-10):.2f}")
