# %%
import concurrent.futures
import multiprocessing
from functools import reduce
from operator import mul
from typing import Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.linalg

# from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import ArrayLike

ArrayList = List[npt.NDArray[np.float64]]
ArrayGenerator = Generator[npt.NDArray[np.float64], None, None]
TTRank = Union[int, Tuple[int, ...]]


def hilbert_tensor(n_dims: int, size: int) -> npt.NDArray:
    """Create a Hilbert tensor of specified size and dimensionality."""
    grid = np.meshgrid(*([np.arange(size)] * n_dims))
    hilbert = 1 / (np.sum(np.array(grid), axis=0) + 1)
    return hilbert


def sqrt_tensor(shape: Tuple[int, ...], a=-0.2, b=2) -> npt.NDArray:
    """Create a tensor of specified shape with square root of a sum a grid.

    Values of grid entries vary between a and b."""

    def sqrt_sum(X):
        return np.sqrt(np.abs(np.sum(X, axis=0)))

    vals = [np.linspace(a, b, s) for s in shape]
    grid = np.stack(np.meshgrid(*vals))
    X = sqrt_sum(grid)
    X /= np.linalg.norm(X)
    return X


def power_decay_tensor(
    shape: Tuple[int], pow: float = 2.0, seed=None
) -> ArrayLike:
    """Create tensor of specified shape such that singular values of each
    unfolding decay with a power law."""
    # if seed is not None:
    #     np.random.seed(np.mod(seed, 2**32 - 1))
    seq = SeedSequence(seed)
    A_seed = seq.generate_state(1)[0]
    A = random_normal(shape=shape, seed=A_seed)
    for mode in range(len(A.shape)):
        A_mat = matricize(A, mode)
        U, S, V = np.linalg.svd(A_mat, full_matrices=False)

        S /= S[0]
        S *= 1 / np.arange(1, len(S) + 1) ** pow
        A_mat = U @ np.diag(S) @ V
        A = dematricize(A_mat, mode, A.shape)
    return A


def matricize(
    A: npt.NDArray, mode: Union[int, Sequence[int]], mat_shape: bool = False
):
    """Matricize tensor ``A`` with respect to ``mode``.

    If mode is an int, return matrix. If mode is a tuple, return tensor of order
    ``len(mode)+1``, unless  ``mat_shape=True``"""
    if isinstance(mode, int):
        mode = (mode,)
    else:  # Try casting to tuple
        mode = tuple(mode)
    perm = mode + tuple(i for i in range(len(A.shape)) if i not in mode)
    A = np.transpose(A, perm)
    right_shape = (np.prod(A.shape[len(mode) :], dtype=int),)
    if mat_shape:
        left_shape = (np.prod(A.shape[: len(mode)], dtype=int),)
    else:
        left_shape = A.shape[: len(mode)]  # type: ignore
    A = A.reshape(left_shape + right_shape)

    return A


def dematricize(A, mode, shape):
    """Undo matricization of ``A`` with respect to ``mode``. Needs ``shape`` of
    original tensor."""
    current_shape = [A.shape[0]] + [s for i, s in enumerate(shape) if i != mode]
    current_shape = tuple(current_shape)
    A = A.reshape(current_shape)
    perm = list(range(1, len(shape)))
    perm = perm[:mode] + [0] + perm[mode:]
    A = np.transpose(A, perm)
    return A


def right_mul_pinv(A, B, cond=None):
    """Compute numerically stable product ``A@np.linalg.pinv(B)``"""
    lstsq = scipy.linalg.lstsq(B.T, A.T, cond=cond)

    return lstsq[0].T


def left_mul_pinv(A, B, cond=None):
    """Compute numerically stable product ``np.linalg.pinv(A)@B``"""
    lstsq = scipy.linalg.lstsq(A, B, cond=cond)

    return lstsq[0]


def projector(X: npt.NDArray, Y: Optional[npt.NDArray] = None) -> npt.NDArray:
    r"""Compute oblique projector :math:`\mathcal P_{X,Y}`"""
    if Y is None:
        Y = X

    P = X @ np.linalg.pinv(Y.T @ X) @ Y.T
    return P


def trim_ranks(
    dims: Tuple[int, ...], ranks: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Return TT-rank to which TT can be exactly reduced

    A tt-rank can never be more than the product of the dimensions on the left
    or right of the rank. Furthermore, any internal edge in the TT cannot have
    rank higher than the product of any two connected supercores. Ranks are
    iteratively reduced  for each edge to satisfy these two requirements until
    the requirements are all satisfied.
    """
    ranks_trimmed = list(ranks)

    for i, r in enumerate(ranks_trimmed):
        dim_left = reduce(mul, dims[: i + 1], 1)
        dim_right = reduce(mul, dims[i + 1 :], 1)
        ranks_trimmed[i] = min(r, dim_left, dim_right)
    changed = True
    ranks_trimmed = [1] + ranks_trimmed + [1]
    for _ in range(100):
        changed = False
        for i, d in enumerate(dims):
            if ranks_trimmed[i + 1] > ranks_trimmed[i] * d:
                changed = True
                ranks_trimmed[i + 1] = ranks_trimmed[i] * d
            if ranks_trimmed[i] > d * ranks_trimmed[i + 1]:
                changed = True
                ranks_trimmed[i] = d * ranks_trimmed[i + 1]
        if not changed:
            break

    return tuple(ranks_trimmed[1:-1])


def process_tt_rank(
    rank: TTRank, shape: Tuple[int, ...], trim: bool
) -> Tuple[int, ...]:
    """
    Process TT rank, and check validity. Makes sure rank is a tuple.

    If ``trim=True``, ranks are trimmed to the smallest possible lossless value.
    """
    # check if rank is iterable, if not use constant rank
    try:
        rank_tuple = tuple(rank)  # type: ignore
    except TypeError:
        rank_tuple = (rank,) * (len(shape) - 1)  # type: ignore
    if len(rank_tuple) != len(shape) - 1:
        raise ValueError(
            f"TT-rank {rank_tuple} doesn't have right number of elements"
        )
    if trim:
        rank_tuple = trim_ranks(shape, rank_tuple)

    return rank_tuple


class MultithreadedRNG:
    """
    Multithreaded standard normal random number generator.

    Copy pasta from numpy docs
    """

    def __init__(self, shape, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        seq = np.random.SeedSequence(seed)
        self._random_generators = [
            np.random.default_rng(s) for s in seq.spawn(threads)
        ]

        self.shape = shape
        n = np.prod(shape)
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(n)
        self.step = np.ceil(n / threads).astype(np.int_)
        self.fill()

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (
                _fill,
                self._random_generators[i],
                self.values,
                i * self.step,
                (i + 1) * self.step,
            )
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        self.values = self.values.reshape(self.shape)

    # def __del__(self):
    #     self.executor.shutdown(False)


def random_normal(shape, seed=None):
    """
    Generate multi-threaded random numbers
    """
    return MultithreadedRNG(shape, seed).values
