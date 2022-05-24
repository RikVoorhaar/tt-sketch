from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from tt_sketch.tensor import SparseTensor


def _Psi_core_slice(
    inds: npt.NDArray[Union[np.int64, np.uint64]],
    entries: npt.NDArray,
    left_sketch_vec: Optional[npt.NDArray],
    right_sketch_vec: Optional[npt.NDArray],
    mu: int,
    j: int,
) -> npt.NDArray[np.float64]:
    """Compute slice of tensor Y[:,j,:] from left+right sketching matrix
    and entries+index of sparse tensor"""
    mask = inds[mu] == j

    if mu == 0:  # only sketch on right
        if right_sketch_vec is None:
            raise ValueError
        Psi_slice = entries[mask] @ right_sketch_vec[:, mask].T
        Psi_slice = Psi_slice.reshape(1, -1)
    elif mu == inds.shape[0] - 1:  # only sketch on left
        if left_sketch_vec is None:
            raise ValueError
        Psi_slice = left_sketch_vec[:, mask] @ entries[mask]
        Psi_slice = Psi_slice.reshape(-1, 1)
    else:  # sketch on both sides
        if left_sketch_vec is None or right_sketch_vec is None:
            raise ValueError
        Psi_slice = (
            left_sketch_vec[:, mask] * entries[mask]
        ) @ right_sketch_vec[:, mask].T
    return Psi_slice


def sketch_omega_sparse(
    left_sketch: npt.NDArray,
    right_sketch: npt.NDArray,
    *,
    tensor: SparseTensor,
    **kwargs,
) -> npt.NDArray:
    return (left_sketch * tensor.entries) @ right_sketch.T


def sketch_psi_sparse(
    left_sketch: npt.NDArray,
    right_sketch: npt.NDArray,
    *,
    tensor: SparseTensor,
    mu: int,
    psi_shape: Tuple[int, int, int],
    **kwargs,
) -> npt.NDArray:
    Psi = np.zeros(psi_shape)
    n_mu = psi_shape[1]
    for j in range(n_mu):
        Psi[:, j, :] = _Psi_core_slice(
            tensor.indices,
            tensor.entries,
            left_sketch,
            right_sketch,
            mu,
            j,
        )
    return Psi
