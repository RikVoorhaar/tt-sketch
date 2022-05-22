# %%
"""In the FasTT experiment they have some interesting experiments with sparse
tensors. Let's first try to recreate the sparse tensor that they described here,
and then see if we can get good results using our algorithm on these tensors."""


# %%
"""First tensor is an image 'Dolphin' reshaped to size 10x20x20x10x15x20x3,
and second is a color video reshaped to 20x18x20x32x12x12x3. Both are then subsampled.

Let's start by loading these in, just like for `frostt`.
Unfortunately, they are on google drive, so manual downloading is necessary."""


from turtle import left
from torch import gather
from tt_sketch.tensor import SparseTensor
import gzip
import shutil
import numpy as np
import pandas as pd
import os
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.tensor import diff_tt_sparse
from time import perf_counter_ns
from tt_sketch.tt_svd import tt_svd
from tt_sketch.utils import trim_ranks
from tt_sketch.tensor import TensorTrain

DOLPHIN_NAME = "data/dolphin-4000x3000.txt.gz"
VIDEO_NAME = "data/video_360x640x144x3.txt.gz"


def load_txt_gz_date(file_name):
    file_unzipped = file_name.split(".gz")[0]
    file_npy = file_unzipped.split(".txt")[0] + ".npy"

    if not os.path.exists(file_npy):
        if not os.path.exists(file_name):
            raise FileNotFoundError(
                "Download the data file first. For dolphin: 'https://drive.google.com/file/d/1cXxqoHhXG3CEBgnzlCXpytHYY-V2XUJ7/view' for video 'https://drive.google.com/file/d/1RvhZmhm7LBVl5tC1iGRj1u5b3ZgBXTNa/view'"
            )
        with gzip.open(file_name, "rb") as f_in:
            with open(file_unzipped, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        data = np.loadtxt(file_unzipped, dtype=np.uint8)
        np.save(file_npy, data, allow_pickle=False)
        os.remove(file_name)
        os.remove(file_unzipped)
    else:
        data = np.load(file_npy)
    return data


def make_sparse_tensor(dense_tensor, sigma=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if sigma is None:
        inds = np.arange(dense_tensor.size)
    else:
        inds = np.sort(
            np.random.choice(
                dense_tensor.size, int(sigma * dense_tensor.size), replace=False
            )
        )
    inds = np.unravel_index(inds, dense_tensor.shape)
    vals = dense_tensor[inds]
    inds = np.stack(inds)

    return SparseTensor(dense_tensor.shape, inds, vals)


def make_dolphin_tensor(sigma, seed=None):
    dolphin_tensor = (
        load_txt_gz_date(DOLPHIN_NAME).reshape(10, 20, 20, 10, 15, 20, 3) / 255
    )
    return make_sparse_tensor(dolphin_tensor, sigma, seed)


def make_video_tensor(sigma, seed=None):
    video_tensor = (
        load_txt_gz_date(VIDEO_NAME).reshape(20, 18, 20, 32, 12, 12, 3) / 255
    )
    return make_sparse_tensor(video_tensor, sigma, seed)


# %%
"""Ok, this thing the algorithm can do very fast compared to FastTT. We should look at how the error improves with higher TT-rank."""

sparse_tensor = make_dolphin_tensor(sigma=0.001)
# full rank, no compression
rank = 100
print("Rank:", rank)
left_rank = trim_ranks(sparse_tensor.shape, (rank,) * (sparse_tensor.ndim - 1))

right_rank = tuple(int(l * 1.5) for l in left_rank)
time = perf_counter_ns()
stt = stream_sketch(sparse_tensor, left_rank, right_rank)
print(f"time taken: {(perf_counter_ns() - time) / 1e9:.2f}s")
tt = stt.to_tt()
error = diff_tt_sparse(tt, sparse_tensor) / sparse_tensor.norm()
print(f"error: {error:.3e}")
# %%
true_tensor = make_dolphin_tensor(sigma=None)
tt.mse_error(true_tensor)
# %%
good_tt = tt_svd(sparse_tensor.to_numpy(), rank=rank)
good_tt = TensorTrain(good_tt)
# %%
print("TT-SVD MSE:", good_tt.mse_error(true_tensor))
error = diff_tt_sparse(good_tt, sparse_tensor) / sparse_tensor.norm()
print("TT-SVD rel error: ", error)
# %%
"""Next is the road network. 
First we need to load it. It's again in a .txt.gz format. This time it's several
lines of comments followed by a tab separated list of numbers. Perhaps we can
export it, and use pandas to load it.

After that, we only keep the first N nodes, that can be done with simple pandas
commands. We then need to apply reverse Cuthill-McKee reordering to the indices,
is that a thing in networkx?


Ok we can just use the scipy version. 

However, it doesn't seem like this is a very good test problem, since we can't
get good low-rank approximations. At the end of the day our method is much
slower, purely because solving the linear systems to obtain the tensor cores is
slow for really high ranks. (we need to solve systems of the size 10^6 x 10x3
== 10x^3, which has costs around 10^12 flops).
"""

# %%
from scipy.sparse import coo_array, eye
from scipy.sparse.csgraph import reverse_cuthill_mckee
import matplotlib.pyplot as plt


def make_road_tensor(half_shape):
    N = np.prod(half_shape)
    shape = half_shape + half_shape
    ROAD_NET_FILE = "data/roadNet-PA.txt.gz"
    # road_net_txt = ROAD_NET_FILE.split(".gz")[0]
    # with gzip.open(ROAD_NET_FILE, "rb") as f_in:
    #     with open(road_net_txt, "wb") as f_out:
    #         shutil.copyfileobj(f_in, f_out)

    road_df = pd.read_csv(
        ROAD_NET_FILE,
        sep="\t",
        header=None,
        skiprows=4,
        names=["from_node", "to_node"],
    )
    road_df = road_df[(road_df["from_node"] < N) & (road_df["to_node"] < N)]

    # Apply reverse Cuthill-McKee reordering
    rows, cols = road_df.values.T
    vals = np.ones(road_df.shape[0])
    mat = coo_array((vals, (rows, cols)), shape=(N, N))
    mat = mat.tocsr()
    permut = reverse_cuthill_mckee(mat)
    mat = mat[permut][:, permut]
    mat = mat.tocoo()

    # convert to sparse tensor
    inds = np.stack(
        np.unravel_index(mat.row, half_shape)
        + np.unravel_index(mat.col, half_shape)
    )
    return SparseTensor(shape, inds, mat.data)


# %%


half_shape = (12,) * 3
sparse_tensor = make_road_tensor(half_shape=half_shape)
# full rank, no compression
left_rank = trim_ranks(
    sparse_tensor.shape, (np.prod(half_shape),) * (sparse_tensor.ndim - 1)
)

right_rank = tuple(int(l * 1.5) for l in left_rank)
time = perf_counter_ns()
stt = stream_sketch(sparse_tensor, left_rank, right_rank)
print(f"time taken: {(perf_counter_ns() - time) / 1e9:.2f}s")

# %%
tt = stt.to_tt()
error = diff_tt_sparse(tt, sparse_tensor) / sparse_tensor.norm()
print(f"error: {error:.3e}")
# %%
"""Alright, next up is a sparse tensor appearing from a finite difference
method. They generate it using some Python code, so that's very convenient.
Let's see if we can get it to work. 

Ok, it's very simple, but really ugly code. I don't want to spend too much time
on this, so I'll just convert it such that it writes straight to an array.

Results of experiment: sketching doesn't really work for these tensors, since
they really don't seem to compress well. I need excessively high rank to get
decent results, but then there's nohing to be gained from sketching in the first
place."""


def make_fdm_tensor(n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def index_to_position(index):
        p = 0
        a, b, c, d, e, f = index
        index = [a, d, b, e, c, f]
        for i in index:
            p = p * n + i
        return p

    p_list = []
    alph_list = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                alpha = np.random.rand(7)
                alpha /= np.linalg.norm(alpha, 1)
                p = index_to_position([i, j, k, i, j, k])
                p_list.append(p)
                alph_list.append(alpha[0])
                if i - 1 >= 0:
                    p = index_to_position([i, j, k, i - 1, j, k])
                    p_list.append(p)
                    alph_list.append(alpha[1])
                if i + 1 < n:
                    p = index_to_position([i, j, k, i + 1, j, k])
                    p_list.append(p)
                    alph_list.append(alpha[2])
                if j - 1 >= 0:
                    p = index_to_position([i, j, k, i, j - 1, k])
                    p_list.append(p)
                    alph_list.append(alpha[3])
                if j + 1 < n:
                    p = index_to_position([i, j, k, i, j + 1, k])
                    p_list.append(p)
                    alph_list.append(alpha[4])
                if k - 1 >= 0:
                    p = index_to_position([i, j, k, i, j, k - 1])
                    p_list.append(p)
                    alph_list.append(alpha[5])
                if k + 1 < n:
                    p = index_to_position([i, j, k, i, j, k + 1])
                    p_list.append(p)
                    alph_list.append(alpha[6])
    inds = np.array(p_list, dtype=np.int64)
    vals = np.array(alph_list, dtype=np.float64)

    inds = np.stack(np.unravel_index(inds, (n,) * 6))
    return SparseTensor((n,) * 6, inds, vals)


# %%
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.tensor import diff_tt_sparse
from time import perf_counter_ns

sparse_tensor = make_fdm_tensor(20)
left_rank = left_rank = (20, 80, 1000, 80, 20)
right_rank = tuple(l + 2 for l in left_rank)
time = perf_counter_ns()
stt = stream_sketch(sparse_tensor, left_rank, right_rank)
print(f"time taken: {(perf_counter_ns() - time) / 1e9:.2f}s")
tt = stt.to_tt()
error = diff_tt_sparse(tt, sparse_tensor) / sparse_tensor.norm()
print(f"error: {error:.3e}")
# %%
