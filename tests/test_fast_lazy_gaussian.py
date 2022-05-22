import sys

import scipy.stats
import numpy as np
import pytest

from tt_sketch.drm.fast_lazy_gaussian import (
    hash_int_c,
    inds_to_sparse_sign,
    inds_to_normal,
    _inds_to_rand_double,
)

# TODO: test cases where r_min > 0


def eight_bytes_to_bits(input_vals):
    """Convert 64-bit arrays to array of bits."""
    shape = input_vals.shape
    vals = np.copy(input_vals).reshape(-1)
    vals = np.frombuffer(vals, dtype=np.uint64)
    all_bytes = []
    for i in range(8):
        bytes_i = (vals << 56 - 8 * i) >> 56
        vals -= bytes_i << 8 * i
        all_bytes.append(bytes_i.astype(np.uint8))

    all_bytes = np.stack(all_bytes)
    bits = np.unpackbits(all_bytes, axis=0)
    bits = bits.reshape((64,) + shape)
    return bits


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("size", (1, 3))
def test_permutation_invariance(n_dims, size):
    """Tests if `inds_to_normal` works as expected"""
    N = 1000
    shape = (10,) * n_dims
    tot_dim = np.prod(shape)
    N = min(N, tot_dim)

    inds_flat = np.random.choice(tot_dim, size=N)
    inds = np.stack(np.unravel_index(inds_flat, shape))
    inds_copy = np.copy(inds)

    shuffle = np.random.permutation(inds.shape[1])
    inds2 = inds[:, shuffle]

    seed = 179
    arr1 = inds_to_normal(inds, shape, 0, size, seed=seed).T
    arr2 = inds_to_normal(inds2, shape, 0, size, seed=seed).T
    assert np.allclose(arr1[:, shuffle], arr2)

    arr3 = inds_to_normal(inds, shape, 0, size, seed=seed + 1).T
    assert not np.allclose(arr1, arr3)

    assert arr1.shape == (size, np.prod(shape))

    # Check if `inds` was changed
    assert np.allclose(inds, inds_copy)


def test_bit_randomness():
    """Test if most bits of hashed_int_c are pretty random"""
    N = 10000
    hashed = np.arange(10000, dtype=np.uint64)
    hash_int_c(hashed)
    hashed_bits = eight_bytes_to_bits(hashed)
    average_deviation = np.mean(hashed_bits, axis=0)
    average_deviation = np.abs(average_deviation - 0.5)
    average_deviation = np.mean(average_deviation)
    # roughly equal chance of being 0 or 1
    assert average_deviation < 0.06

    # roughly 50/50 chance of being different from neighbor
    average_deviation2 = np.mean(
        np.abs(np.diff(hashed_bits.astype(float), axis=1)), axis=0
    )
    average_deviation2 = np.abs(average_deviation2 - 0.5)
    average_deviation2 = np.mean(average_deviation2)
    assert average_deviation2 < 0.06


def test_inds_to_sparse_sign():
    shape = (10, 12, 14, 7)
    tot_dim = np.prod(shape, dtype=int)
    N = tot_dim
    np.random.seed(179)
    inds_flat = np.random.choice(tot_dim, size=N)
    inds_flat = np.arange(tot_dim)
    inds = np.array(np.unravel_index(inds_flat, shape))
    seed = 5
    rank = 100
    nonzero_per_row = 20

    nums = np.array(
        inds_to_sparse_sign(inds, shape, rank, 0, rank, nonzero_per_row, seed)
    )

    assert nums.shape == (tot_dim, rank)

    # Same chance of having non-zero element in each column
    mean_deviation = np.mean(
        np.abs(np.mean(np.abs(nums), axis=0) - nonzero_per_row / rank)
    )
    assert mean_deviation < 0.01

    # number of nonzero per row is correct
    assert np.all(np.sum(np.abs(nums), axis=1) == nonzero_per_row)

    # only +/-1
    assert np.all(np.unique(nums) == np.array([-1, 0, 1]))

    rank_min = 5
    rank_max = 20
    nums2 = np.array(
        inds_to_sparse_sign(
            inds, shape, rank, rank_min, rank_max, nonzero_per_row, seed
        )
    )
    assert np.all(nums[:, rank_min:rank_max] == nums2)


def test_inds_to_rand_double():
    """Test uniform distribution of mantissa and exponent"""
    shape = (10, 12, 14, 7)
    rank = 17
    tot_dim = np.prod(shape, dtype=int)
    N = tot_dim
    inds_flat = np.random.choice(tot_dim, size=N)
    inds_flat = np.arange(tot_dim)
    inds = np.array(np.unravel_index(inds_flat, shape))
    inds = inds.astype(np.uint64)
    seed = 5
    shape_arr = np.array(shape, dtype=np.uint64)
    inds_copy = np.copy(inds)

    rank_min = rank - 5
    rand_nums = np.array(
        _inds_to_rand_double(inds_copy, shape_arr, 0, rank, seed)
    )
    assert np.all(inds == inds_copy)  # function is not allowed to edit inds

    # Test rand_min
    rand_nums2 = np.array(
        _inds_to_rand_double(inds, shape_arr, rank_min, rank, seed)
    ).reshape(N, rank - rank_min)
    assert np.all(
        rand_nums.reshape(N, rank)[:, rank_min:] == rand_nums2
    )  # function
    mant, exp = np.frexp(rand_nums)

    # Check mantissa is always in [0.5,1]
    assert np.min(mant) >= 0.5
    assert np.max(mant) <= 1

    # Check the exponent is in the expected range
    assert np.min(exp) >= -510
    assert np.max(exp) <= 1

    # some statistical tests
    uniform_dev = np.sort(mant) - np.linspace(0.5, 1, tot_dim * rank)
    assert np.mean(np.abs(uniform_dev)) < 0.001
    assert (
        np.mean(np.abs(np.sort(exp) - np.linspace(-510, 1, tot_dim * rank))) < 2
    )

    # Test if increasing rank leaves first columns invariant
    rank2 = rank + 5
    rand_nums1 = np.reshape(rand_nums, (N, rank))
    rand_nums2 = np.array(
        _inds_to_rand_double(inds_copy, shape_arr, 0, rank2, seed)
    )
    rand_nums2 = np.reshape(rand_nums2, (N, rank2))
    assert np.all(rand_nums2[:, :rank] == rand_nums1)


def test_inds_to_normal():
    shape = (10, 12, 14, 7)
    rank = 17
    tot_dim = np.prod(shape, dtype=int)
    N = tot_dim
    inds_flat = np.random.choice(tot_dim, size=N)
    inds_flat = np.arange(tot_dim)
    inds = np.array(np.unravel_index(inds_flat, shape))
    inds = inds.astype(np.uint64)
    seed = 5
    shape_arr = np.array(shape, dtype=np.uint64)
    rand_nums = np.array(inds_to_normal(inds, shape_arr, 0, rank, seed))
    dist = np.sort(scipy.stats.norm.cdf(rand_nums.reshape(-1)))
    deviation = np.mean(np.abs(dist - np.linspace(0, 1, len(dist))))
    assert deviation < 0.01

    rank_min = 5
    rand_nums1 = rand_nums.reshape(N, rank)
    rand_nums2 = np.array(
        inds_to_normal(inds, shape_arr, rank_min, rank, seed)
    ).reshape(N, rank - rank_min)
    assert np.all(rand_nums1[:,rank_min:] == rand_nums2)
