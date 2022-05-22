# %%
from tt_sketch.tensor import DenseTensor, TensorTrain
from tt_sketch.utils import projector, matricize
from tt_sketch.drm import DenseGaussianDRM
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def hilbert_tensor(n_dims: int, size: int) -> DenseTensor:
    grid = np.meshgrid(*([np.arange(size)] * n_dims))
    hilbert = 1 / (np.sum(np.array(grid), axis=0) + 1)
    return DenseTensor((size,) * n_dims, hilbert)


n_dims = 5
dim = 8
r = 5
l = r
tensor = hilbert_tensor(n_dims, dim)
tensor_np = tensor.to_numpy()

left_rank = (r,) * (n_dims - 1)
right_rank = (r + l,) * (n_dims - 1)
left_drm = DenseGaussianDRM(left_rank, tensor.shape, transpose=False)
right_drm = DenseGaussianDRM(right_rank, tensor.shape, transpose=True)
X_list = right_drm.sketch_dense(tensor)
Y_list = left_drm.sketch_dense(tensor)

# projector(X_list[n_dims-2], Y_list[n_dims-2])
def TX_Y(mu):
    TX = matricize(tensor_np, range(mu + 1), mat_shape=True) @ X_list[mu].T
    Y = Y_list[mu].T
    return TX, Y


def proj_mu(mu):
    TX, Y = TX_Y(mu)
    return projector(TX, Y)


approx = proj_mu(n_dims - 2) @ matricize(
    tensor_np, range(n_dims - 1), mat_shape=True
)
approx = tensor_np - approx.reshape(tensor.shape)
approx_norm = np.linalg.norm(approx)
print(f"Approximation norm: {approx_norm:.4e}")


for mu in range(n_dims - 3, -1, -1):
    print("-" * 10)
    approx_mat = matricize(approx, range(mu + 1), mat_shape=True)
    approx_proj = proj_mu(mu) @ approx_mat
    norm_increase = np.linalg.norm(approx_proj) / np.linalg.norm(approx)
    spectral_norm = np.linalg.norm(proj_mu(mu), ord=2)
    print(f"{mu=}\n{norm_increase=} {spectral_norm=}")

    TX, Y = TX_Y(mu)
    first_approx = np.linalg.norm(TX @ np.linalg.pinv(Y.T @ TX), ord=2)
    print(Y.shape, approx.shape)
    first_approx *= np.linalg.norm(Y.T @ approx_mat) / np.linalg.norm(approx_proj)
    print(f"{first_approx=}")

    random_mat = np.random.normal(size=approx_proj.shape)
    random_norm_increase = np.linalg.norm(
        proj_mu(mu) @ random_mat
    ) / np.linalg.norm(random_mat)
    print(f"{random_norm_increase=}")
    first_approx_random = np.linalg.norm(
        TX @ np.linalg.pinv(Y.T @ TX), ord=2
    ) * np.linalg.norm(Y.T @ random_mat) / np.linalg.norm(random_mat)
    print(f"{first_approx_random=}")

    approx = approx_proj.reshape(tensor.shape)

# %%
"""What does orthogonal projector do to pseudo inverse product?"""

n = 10
r = 5
l = 5
Q = np.random.normal(size=(n,l))
Q = np.linalg.qr(Q)[0]
Pi = Q@Q.T

M = np.random.normal(size=(n,r))

print(np.allclose(np.linalg.pinv(M)@M,np.eye(r)))
np.linalg.pinv(Pi@M)@M

X = np.random.normal(size=(r,n))
X[:] = 0
print(np.allclose(Pi@M@(np.linalg.pinv(Pi@M))@M, Pi@M))

pm_proj = Pi@M@np.linalg.pinv(Pi@M)
left = Pi-pm_proj
right = Pi@(np.eye(n)-pm_proj)@(np.eye(n)-M@X)
np.allclose(left,right)
# %%
"""How does a random gaussian matrix affect the spectrum / condition number of a
matrix?"""

# First generate a random matrix with a particular spectrum, e.g. power law

(m, n, r) = (15, 10, 5)
k = min(m, n)

A = np.random.normal(size=(m, n))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S = S / S[0]
# S = S * np.logspace(0, -5, k)  # Exponential decay
# S = -np.sort(-(2**(10*np.random.normal(size=k))))
# S = S / S[0]
S = S * 1 / (2 ** np.linspace(1, 10, k))  # Power law

A = np.diag(S)

X = np.random.normal(size=(k, r))
AX = A @ X
AX_svdvals = scipy.linalg.svdvals(AX)
plt.plot(AX_svdvals)
# plt.plot(S[: 2 * r])
plt.plot(S)
plt.yscale("log")

avg_cond = []
spect_X = []
for _ in range(30):
    X = np.random.normal(size=(k, r))
    AX = A @ X
    avg_cond.append(np.linalg.cond(AX))
    spect_X.append(np.linalg.norm(X, ord=2))
avg_cond = np.array(avg_cond)
spect_X = np.array(spect_X)
np.mean(avg_cond) / (np.mean(spect_X) * S[0] / S[r]), np.mean(avg_cond) / (
    S[0] / S[r]
)

# %%
"""What is the norm and singular values of a TT?"""

mean_norms = []
dims = np.arange(5,20)
dimsum = []
for d in dims:
    shape = (d, d+1, d+2, d-3)
    dimsum.append(np.sum(np.array(shape))**2)
    n_dims = len(shape)
    rank = tuple(range(3,3+n_dims-1))
    norms = []
    for _ in range(20):
        tt = TensorTrain.random(shape, rank)
        norms.append(tt.norm())
    mean_norms.append(np.mean(norms))

dimsum = np.array(dimsum)
mean_norms = np.array(mean_norms)
plt.plot(dims, mean_norms/dimsum)

# %%
