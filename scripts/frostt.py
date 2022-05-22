# %%
from operator import sub
from data_download_util import download_data
import os.path
from tqdm import tqdm
import numpy as np
import gzip
from tt_sketch.tensor import SparseTensor
from tt_sketch.sketched_tensor_train import SketchedTensorTrain
from tt_sketch.drm import (
    TensorTrainDRM,
    SparseSignDRM,
    SparseGaussianDRM,
)
from tt_sketch.sketching_methods.sparse_sketch import (
    sparse_sketch,
)

# File url for the NIPS dataset
nips_tensor = {
    "file_url": (
        "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz"
    ),
    "nnz": 3101609,
    "shape": (2482, 2862, 14036, 17),
}
uber_tensor = {
    "file_url": (
        "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz"
    ),
    "nnz": 3309490,
    "shape": (183, 24, 1140, 1717),
}
lbnl_tensor = {
    "file_url": (
        "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/lbnl-network/lbnl-network.tns.gz"
    ),
    "nnz": 1698825,
    "shape": (1605, 4198, 1631, 4209, 868131),
}

matmul_tensor = {
    "file_url": (
        "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/matrix-multiplication/matmul_5-5-5.tns.gz"
    ),
    "nnz": 5 * 5 * 5,
    "shape": (5 * 5, 5 * 5, 5 * 5),
}


def process_frostt_tensor(filepath, nnz, shape):
    with gzip.open(filepath) as f:
        lines = f.readlines()
        num_lines = nnz
        dim = len(lines[0].split(b" ")) - 1
        indices = np.zeros((num_lines, dim), dtype=np.int64)
        entires = np.zeros(num_lines, dtype=np.float64)
        for i, line in tqdm(
            enumerate(lines), total=nnz, desc="Processing file"
        ):
            split = line.split(b" ")
            indices[i] = [int(s) - 1 for s in split[:-1]]  # indices are 1-based
            entires[i] = float(split[-1])
    tensor = SparseTensor(shape, indices.T, entires)
    return tensor


def get_frostt_tensor(file_url, nnz, shape):
    filename = file_url.split("/")[-1]
    filepath = os.path.join("data", filename)
    npzpath = filepath.split(".gz")[0] + ".npz"
    try:
        npz_load = np.load(npzpath)
        tensor = SparseTensor(
            tuple(npz_load["shape"]), npz_load["indices"], npz_load["entries"]
        )
        return tensor
    except FileNotFoundError:
        download_data(file_url)
        tensor = process_frostt_tensor(filepath, nnz, shape)
        indices = tensor.indices
        entries = tensor.entries
        print("Saving processed file")
        np.savez_compressed(
            npzpath, indices=indices, entries=entries, shape=shape
        )
        print("Done!")
        return tensor


# %%
tensor_info = matmul_tensor
tensor = get_frostt_tensor(
    tensor_info["file_url"], tensor_info["nnz"], tensor_info["shape"]
)
ranks = range(2, 50)
errors = []


def compute_error(rank):
    Psi_cores, Omega_mats = sparse_sketch(
        tensor,
        rank,
        rank + 1,
        SparseGaussianDRM,
        SparseGaussianDRM,
        seed=179,
        pbar=None
    )
    stt = SketchedTensorTrain(None, None, Psi_cores, Omega_mats)
    tt = stt.to_tt()
    sample = np.random.choice(tensor.nnz, size=10000)
    inds = tensor.indices[:, sample]
    entr = tensor.entries[sample]

    rel_error = np.linalg.norm(tt.gather(inds) - entr) / np.linalg.norm(entr)
    # errors.append(rel_error)
    return rel_error


for rank in tqdm(ranks):
    errors.append(compute_error(rank))

# %%
import matplotlib.pyplot as plt

plt.plot(ranks, errors)
plt.yscale("log")

# %%
