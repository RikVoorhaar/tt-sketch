import numpy as np
import numpy.typing as npt
from tt_sketch.utils import right_mul_pinv
from tt_sketch.drm import TensorTrainDRM


def orth_step(
    Psi: npt.NDArray[np.float64], Omega: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    Psi_shape = Psi.shape
    Psi_mat = Psi.reshape((Psi_shape[0] * Psi_shape[1], Psi_shape[2]))
    Psi_mat = right_mul_pinv(Psi_mat, Omega)
    Psi_mat, _ = np.linalg.qr(Psi_mat)
    Psi = Psi_mat.reshape(Psi_shape[0], Psi_shape[1], Omega.shape[0])
    return Psi


class OrthogTTDRM:
    def __init__(self, rank, tensor, sketch_method_name):
        self.rank = rank
        self.drm = TensorTrainDRM(rank, tensor.shape, transpose=False, cores=[])
        self.generator = None
        self.tensor = tensor
        self.sketch_method = getattr(self.drm, sketch_method_name)

    def add_core(self, core):
        self.drm.cores.append(core)
        if self.generator is None:
            self.generator = self.sketch_method(self.tensor)

    def __next__(self):
        return next(self.generator)
