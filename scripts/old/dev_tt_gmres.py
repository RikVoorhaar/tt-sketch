# %%
import numpy as np
import matplotlib.pyplot as plt
from tt_sketch.tensor import TensorTrain
from tt_sketch.tt_gmres import MPO, tt_gmres

shape = (3, 3, 3)
mpo = MPO.random(4, shape, shape)
# mpo = MPO.eye(shape)
tt_true = TensorTrain.random(shape, 2)
tt = mpo(tt_true)

rank = 6
tt_approx, history = tt_gmres(mpo, tt, rank, maxiter=100,tolerance=1e-8)
print(f"relative error: {mpo(tt_approx).mse_error(tt) / tt.norm():.4e}")

# tt_approx, _ = tt_gmres(mpo, tt, rank, maxiter=20,tolerance=0,x0=tt_approx)
# print(f"relative error: {mpo(tt_approx).error(tt) / tt.norm():.4e}")

# tt_approx, _ = tt_gmres(mpo, tt, rank, maxiter=20,tolerance=0,x0=tt_approx)
# print(f"relative error: {mpo(tt_approx).error(tt) / tt.norm():.4e}")
# H
# # %%
# # Let's check the condition mpo(nu[k-1]) = sum(nu[j]*h_jk) for j=0,...,k
# # wait, no, for that we need nu_list, which is not returned by history. Well we
# # can of course just return it for debug purposes.

# k = 4
# nu_list = history["nu_list"]
# rhs = 0
# for j in range(k + 2):
#     rhs += H[j, k] * nu_list[j].to_numpy()

# lhs = mpo(nu_list[k]).to_numpy()
# rhs - lhs
# np.linalg.norm(rhs - lhs), np.linalg.norm(lhs), np.linalg.norm(rhs)
# %%
# np.linalg.norm(nu_list[1].to_numpy())
plt.yscale("log")
plt.plot(history["true_residual_norm"][1:])
# %% What kind of condition number does our random MPO have? Preferably its not
# too far from 1.


def mpo_to_mat(mpo):
    mpo_dense = mpo.to_numpy()
    in_dim = np.prod(mpo.in_shape)
    out_dim = np.prod(mpo.out_shape)
    permut = tuple(range(0, mpo.ndim * 2, 2)) + tuple(range(1, mpo.ndim * 2, 2))
    return mpo_dense.transpose(permut).reshape(in_dim, out_dim)


# shape = (3, 4, 5)
# mpo = MPO.random(2, shape, shape)
# mpo_dense = mpo.to_numpy()

# tot_dim = np.prod(shape)
# mpo_mat = mpo_dense.transpose((0, 2, 4, 1, 3, 5)).reshape(tot_dim, tot_dim)
mpo_mat = mpo_to_mat(mpo)
_, S, _ = np.linalg.svd(mpo_mat)
np.linalg.cond(mpo_mat)  # not too bad...

# so the MPOs are relatively well conditioned. What happens if we try to solve
# the linear problem with dense matrices?

# %%
ndim = 3
# %%

shape = (6, 6, 6)
mpo = MPO.random(4, shape, shape)
# mpo = MPO.eye(shape)
tt_true = TensorTrain.random(shape, 2)
tt = mpo(tt_true)

mpo_mat = mpo_to_mat(mpo)
tt_dense = tt.to_numpy()
tt_vec = tt_dense.reshape(-1)

tt_vec_estim, _, _, _ = np.linalg.lstsq(mpo_mat, tt_vec, rcond=None)
assert np.linalg.norm(mpo_mat @ tt_vec_estim - tt_vec) < 1e-8
tt_estim = TensorTrain.from_dense(tt_vec_estim.reshape(shape))
assert mpo(tt_estim).mse_error(tt) < 1e-12

# we can solve the dense version of the problem just fine.

# %%
from tt_sketch.tt_gmres import tt_weighted_sum, tt_weighted_sum_sketched

n_elems = 10
x0 = TensorTrain.random(shape,3)
print(x0.norm())
tt_list = [TensorTrain.random(shape,3) for _ in range(n_elems)]
coeffs = np.random.normal(size=n_elems)

dense = x0.to_numpy()
for tt,coeff in zip(tt_list,coeffs):
    dense+= coeff*tt.to_numpy()


x0 = tt_weighted_sum(x0,coeffs,tt_list,1e-6,6)
np.linalg.norm(x0.to_numpy() - dense)
# %%

x0.norm()