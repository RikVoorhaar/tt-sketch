# %%
"""Compress high-rank CP obtained from random forest to TT

We can write a decision tree as a low-rank CP. Since random forests are just
weighted sums of decision trees, they too can be expressed as a CP. To limit the
dimension of the resulting tensor we do need to compress the dimension of this
CP by taking a subset of the thresholds.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ttml.forest_compression import compress_forest_thresholds, forest_to_CP
import numpy as np
from tt_sketch.tensor import CPTensor
from tt_sketch.sketch import stream_sketch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from data_download_util import download_data

FILE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291"
    "/airfoil_self_noise.dat"
)
download_data(FILE_URL)
df = pd.read_csv("data/airfoil_self_noise.dat", sep="\t", header=None)
Xy = df.to_numpy()
print(Xy.shape)

scaler = StandardScaler()
Xy = scaler.fit_transform(Xy)

y = Xy[:, 5]
X = Xy[:, :5]
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=179,
)
forest = RandomForestRegressor(
    max_leaf_nodes=500, max_depth=None, n_estimators=50
)
forest.fit(X_train, y_train)


def mse(estim):
    return np.mean((estim.predict(X_val) - y_val) ** 2)


uncompressed_loss = mse(forest)
uncompressed_loss

# %%
num_thresholds = np.arange(25, 100, 5)
errors = []
for num_t in tqdm(num_thresholds):
    c_forest, _ = compress_forest_thresholds(forest, num_t)
    errors.append(mse(c_forest))

plt.plot(num_thresholds, errors, ".")
plt.axhline(uncompressed_loss, ls="--", c="k")

# %%
"""40 seems like a good number of thresholds for this forest"""
c_forest, thresh = compress_forest_thresholds(forest, 10)
cp_cores = forest_to_CP(c_forest, thresh, take_mean=True)
cp_tensor = CPTensor(cp_cores)

N = 1000
dim = np.prod(cp_tensor.shape)
dense_idx = np.random.choice(dim, size=N)
idx = np.unravel_index(dense_idx, cp_tensor.shape)
cp_samples = cp_tensor.gather(idx)

ranks = range(5, 50, 1)
errors = []
for rank in tqdm(ranks):
    stt = stream_sketch(cp_tensor, left_rank=rank, right_rank=rank*2)
    tt = stt.to_tt()
    error = np.linalg.norm(cp_samples - tt.gather(idx)) / np.linalg.norm(
        cp_samples
    )
    errors.append(error)

# %%

plt.plot(ranks, errors)
plt.yscale("log")

"""Seems like it works, but the errors are just way too high. That kinda shows
that this can't be compressed well as a tt, which should not come as a
surprise..."""
# %%
