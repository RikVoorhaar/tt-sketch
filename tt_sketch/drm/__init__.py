from tt_sketch.drm.dense_gaussian_drm import DenseGaussianDRM
from tt_sketch.drm.sparse_gaussian_drm import SparseGaussianDRM
from tt_sketch.drm.tensor_train_drm import TensorTrainDRM
from tt_sketch.drm.sparse_sign_drm import SparseSignDRM


ALL_DRM = (DenseGaussianDRM, SparseGaussianDRM, TensorTrainDRM, SparseSignDRM)
