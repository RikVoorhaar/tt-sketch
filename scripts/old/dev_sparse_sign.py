# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

r = 100
z = 10
N = int(1e4)

v = np.zeros((N, r), dtype=int)
v[:, :z] = np.random.randint(2, size=(N, z)) * 2 - 1

rand_swaps = [np.random.randint(i, r, size=N) for i in range(z)]

for i in tqdm(range(N)):
    for j in range(z):
        temp = v[i, j]
        v[i, j] = v[i, rand_swaps[j][i]]
        v[i, rand_swaps[j][i]] = temp


def eight_bytes_to_bits(input_vals):
    vals = np.copy(input_vals)
    all_bytes = []
    for i in range(8):
        bytes_i = (vals << 56 - 8 * i) >> 56
        vals -= bytes_i << 8 * i
        all_bytes.append(bytes_i.astype(np.uint8))

    all_bytes = np.stack(all_bytes)
    bits = np.unpackbits(all_bytes, axis=0)
    return bits


# %%
from tt_sketch.drm.fast_lazy_gaussian import inds_to_sparse_sign

shape = (10, 10, 10, 10)
tot_dim = np.prod(shape, dtype=int)
N = tot_dim
inds_flat = np.random.choice(tot_dim, size=N)
inds_flat = np.arange(tot_dim)
inds = np.array(np.unravel_index(inds_flat, shape))
# inds = inds.astype(np.uint64)
seed = 5
# shape_arr = np.array(shape, dtype=np.uint64)
rank = 100
nonzero_per_row = 20

nums = np.array(inds_to_sparse_sign(inds, shape, rank, nonzero_per_row, seed))

np.mean(np.abs(np.mean(np.abs(nums), axis=0) - nonzero_per_row / rank))
np.all(np.sum(np.abs(nums), axis=1) == nonzero_per_row)
np.all(np.unique(nums) == np.array([-1, 0, 1]))

# %%
from tt_sketch.drm.fast_lazy_gaussian import _inds_to_rand_double

shape = (10, 10, 10, 10)
tot_dim = np.prod(shape, dtype=int)
N = tot_dim
inds_flat = np.random.choice(tot_dim, size=N)
inds_flat = np.arange(tot_dim)
inds = np.array(np.unravel_index(inds_flat, shape))
inds = inds.astype(np.uint64)
seed = 5
shape_arr = np.array(shape, dtype=np.uint64)
rand_nums = np.array(_inds_to_rand_double(inds, shape_arr, rank, seed))
mant, exp = np.frexp(rand_nums)
assert np.min(mant) >= 0.5
assert np.max(mant) <= 1
uniform_dev = np.sort(mant) - np.linspace(0.5, 1, tot_dim * rank)
assert np.mean(np.abs(uniform_dev)) < 0.001
assert np.min(exp) >= -510
assert np.max(exp) <= 1
assert (
    np.mean(np.abs(np.sort(exp) - np.linspace(-510, 1, tot_dim * rank))) < 0.5
)

# %%
from tt_sketch.drm.fast_lazy_gaussian import inds_to_normal
import scipy.stats


shape = (10, 10, 10, 10)
tot_dim = np.prod(shape, dtype=int)
N = tot_dim
inds_flat = np.random.choice(tot_dim, size=N)
inds_flat = np.arange(tot_dim)
inds = np.array(np.unravel_index(inds_flat, shape))
inds = inds.astype(np.uint64)
seed = 5
shape_arr = np.array(shape, dtype=np.uint64)
rand_nums = np.array(inds_to_normal(inds, shape_arr, rank, seed))
dist = np.sort(scipy.stats.norm.cdf(rand_nums.reshape(-1)))
deviation = np.mean(np.abs(dist-np.linspace(0,1,len(dist))))
assert deviation < 0.01
# (rand_nums)
# rand_nums
# %%
eight_bytes_to_bits(nums.reshape(-1))
# %%
from tt_sketch.drm.fast_lazy_gaussian import hash_int_c


hashed = np.arange(10000, dtype=np.uint64)
hashed = hashed + 0x48AB81DE7328A32F
hash_int_c(hashed)
hashed_bits = eight_bytes_to_bits(hashed)
average_deviation = np.mean(hashed_bits, axis=0)
average_deviation = np.abs(average_deviation - 0.5)
average_deviation = np.mean(average_deviation)
print(average_deviation)

average_deviation2 = np.mean(
    np.abs(np.diff(hashed_bits.astype(float), axis=1)), axis=0
)
average_deviation2 = np.abs(average_deviation2 - 0.5)
average_deviation2 = np.mean(average_deviation2)


# %%
np.sum(np.abs(nums.reshape(N, rank)), axis=0)
# %%
plt.plot(
    np.sort(np.mean(np.abs(nums).reshape(N, rank), axis=0))
    * rank
    / nonzero_per_row
)
# %%
x = mantissa * 2 - 1
np.max(x), np.min(x)
# %%
plt.plot(np.sort(mantissa.reshape(-1)))
# _inds_to_sparse_sign()
# %%
from tt_sketch.drm.fast_lazy_gaussian import hash_int_c

inds_flat = np.arange(tot_dim)
hashed = inds_flat.astype(np.uint64)  # + 0X4BE98134A5976FD3
hash_int_c(hashed)
hashed = np.bitwise_or(hashed, 0x2000000000000000)
hashed = np.bitwise_and(hashed, 0x3FFFFFFFFFFFFFFF)
floats = np.frombuffer(hashed, dtype=np.float64)
print("num nan:", np.sum(np.isnan(floats)))
# np.mean(eight_bytes_to_bits(hashed),axis=0)[:10]
mantissa, exponent = np.frexp(floats)
np.min(mantissa)
# plt.plot(np.sort(mantissa))
#%%
np.unpackbits(np.array(0x20, dtype=np.uint8))
# %%
inds = np.random.choice(len(hashed), size=64, replace=False)
singular = eight_bytes_to_bits(hashed[np.isnan(floats)])
np.mean(singular, axis=1)
np.frexp(floats[np.isnan(floats)])
np.frexp(floats)
# %%
np.log2(np.sum(np.isnan(nums)) / len(hashed))
# %%
A = np.array([10, 20])
np.frombuffer(A)

MAX_INT64 = np.iinfo(np.int64).max
MIN_INT64 = np.iinfo(np.int64).min
rnd = np.random.randint(MIN_INT64, MAX_INT64, size=10000)

mantissa, exponent = np.frexp(np.frombuffer(rnd, dtype=np.float64))
plt.plot(np.sort(exponent))

# %%
def hash_int_py(vals):
    """Use simple hashing for generating random numbers from indices
    deterministically. See
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    N = vals.shape[0]
    shift1 = 30
    shift2 = 27
    shift3 = 31
    mult1 = 0xBF58476D1CE4E5B9
    mult2 = 0x94D049BB133111EB
    add1 = 0x4BE98134A5976FD3

    vals += add1
    vals ^= vals >> shift1
    vals *= mult1
    vals ^= vals >> shift2
    vals *= mult2
    vals ^= vals >> shift3
    # vals[:] = np.mod(vals, 2 ** 32 - 1)


# %%
from tt_sketch.drm.fast_lazy_gaussian import hash_int_c

input_vals = np.arange(int(1e6), dtype=np.uint64)
input_vals += np.random.randint(0, MAX_INT64, dtype=np.uint64)
print(input_vals.dtype)
# input_vals >> 10
i_vals1 = input_vals.copy()
i_vals2 = input_vals.copy()
hash_int_py(i_vals1)
hash_int_c(i_vals2)
shift = 32
diffs = (i_vals1 >> 0) - i_vals2
diffs = (i_vals1 - i_vals2) >> shift
plt.plot(np.sort(diffs))

# i_vals1>>0
# %%
input_vals = np.arange(int(1e6), dtype=np.uint64)
hash_int_c(input_vals)
mantissa, exponent = np.frexp(input_vals)
# plt.plot(np.sort(exponent))

rnd = np.random.randint(MIN_INT64, MAX_INT64, size=1000000)
rnd = np.frombuffer(rnd, np.uint64)
# input_vals = rnd
# byte0 = (input_vals << 48) >> 48
# input_vals -= byte0
# byte1 = (input_vals << 32) >> 48
# input_vals -= byte1 << 16
# byte2 = (input_vals << 16) >> 48
# input_vals -= byte2 << 32
# byte3 = input_vals >> 48

# bytes = np.stack([byte0, byte1, byte2, byte3]).astype(np.uint8)

# bits = np.unpackbits(bytes.T, axis=1)
means = np.mean(bits, axis=1)
plt.plot(means)

# %%
0xBF58476D1CE4E5B9 / np.iinfo(np.uint64).max

# %%
all_bytes = []
vals = np.copy(rnd)
for i in range(8):
    bytes_i = (vals << 56 - 8 * i) >> 56
    vals -= bytes_i << 8 * i
    all_bytes.append(bytes_i.astype(np.uint8))

all_bytes = np.stack(all_bytes)
bits = np.unpackbits(all_bytes, axis=0)
means = np.mean(bits, axis=0)
plt.plot(np.sort(means))
# %%
N = 1e7
# rnd = np.random.randint(0, 256, size=(int(N),1), dtype=np.uint8)
rnd = np.arange(256, dtype=np.uint8).reshape(-1, 1)
bits = np.unpackbits(rnd, axis=1)
means = np.mean(bits, axis=0)
plt.plot(np.sort(means))
# np.std(np.bincount(np.sort(rnd.reshape(-1)))) / np.sqrt(N)
np.mean(bits[:, -6])
