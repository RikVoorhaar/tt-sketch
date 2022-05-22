# %%
import numpy as np
import matplotlib.pyplot as plt


m = 17
n = 12


norms1 = []
norms2 = []
norms3 = []
norms4 = []
norms5 = []
norms6 = []
norms7 = []
for _ in range(20):
    A = np.random.uniform(size=(n, n))
    # A = np.eye(n)
    B = np.random.normal(size=(m, m))
    B = np.eye(m)
    for _ in range(100):
        X = np.random.normal(size=(m, n))
        norms1.append(np.linalg.norm(A @ np.linalg.pinv(X) @ B) ** 2)
        norms2.append(np.linalg.norm(A) ** 2)

        _, S1, _ = np.linalg.svd(A)
        _, S2, _ = np.linalg.svd(B)
        k = min(n, m)
        norms3.append(np.dot(S1[:k], S2[:k]))

        norms4.append(
            np.linalg.norm(np.diag(S1) @ np.linalg.pinv(np.diag(S2**-1) @ X))
            ** 2
        )

        norms5.append(
            np.linalg.norm(np.linalg.pinv(X @ np.diag(S1**-1)) @ np.diag(S2))
            ** 2
        )

        norms6.append(
            np.sum(
                np.diag(
                    np.diag(S2)
                    @ np.linalg.pinv(X @ np.diag(S1**-2) @ X.T)
                    @ np.diag(S2)
                )
            )
        )

        norms7.append(
            np.sum(
                np.diag(
                    np.diag(S1)
                    @ np.linalg.pinv(X.T @ np.diag(S2**-2) @ X)
                    @ np.diag(S1)
                )
            )
        )


norms1 = np.array(norms1)
norms2 = np.array(norms2)
norms3 = np.array(norms3)
pred_frac = 1 / (np.abs(m - n) - 1)
print(np.mean(norms2 / norms1))
print(np.mean(norms2) / np.mean(norms1))
print(np.mean(norms3 / norms1))
print(np.mean(norms3) / np.mean(norms1))
print("-" * 10)
print(np.mean(norms1) / np.mean(norms4))
print(np.mean(norms1) / np.mean(norms5))
print(np.mean(norms1) / np.mean(norms6))
print(np.mean(norms1) / np.mean(norms7))
print(1 / pred_frac)

# %%

r = 5
n = 10
m = 20
res_list = []
for _ in range(1000):
    Yt = np.random.normal(size=(r, m))
    Q = np.random.normal(size=(m, n))
    Q, _ = np.linalg.qr(Q)
    res_list.append((Yt @ Q).reshape(-1))
res = np.stack(res_list)
np.mean(res, axis=0), np.std(res, axis=0)

# %%

N = 100
M = 50
r = 10
l = 4

A = np.random.normal(size=(N, r))
Yt = np.random.normal(size=(r + l, N))
B = np.random.normal(size=(r + l, M))
Q, _ = np.linalg.qr(A)

np.linalg.norm(A @ np.linalg.pinv(Yt @ A) @ B) ** 2, np.linalg.norm(
    np.linalg.pinv(Yt @ Q) @ B
) ** 2

# %%

res1 = []
res2 = []
for _ in range(100):
    A = np.random.normal(size=(N, M))
    Yt = np.random.normal(size=(r, N))
    res1.append(np.linalg.norm(A) ** 2 * r)
    res2.append(np.linalg.norm(Yt @ A) ** 2)

np.mean(res1) / np.mean(res2)

# %%
N = 100
m = 20
n = 10
Q = np.random.normal(size=(N, m))
Q, _ = np.linalg.qr(Q)
A = np.random.normal(size=(m, n))
np.linalg.norm(Q @ A) - np.linalg.norm(A)


# %%
from tt_sketch.utils import projector

N = 100
m = 20
r = 10
l = 5
Y = np.random.normal(size=(N, r))
X = np.random.normal(size=(N, r + l))
A = np.random.normal(size=(N, m))

Q, _ = np.linalg.qr(X)
np.linalg.norm(projector(X, Y) - projector(Q, Y))

# %%
r = 10
l = 5

R = np.random.normal(size=(r + l, r + l))
R = np.triu(R)
X = np.random.normal(size=(r, r + l))
U, S, Vt = np.linalg.svd(R)

print("||R(XR)^+||_2 = ", np.linalg.norm(R @ np.linalg.pinv(X @ R), ord=2))
print(
    "||SVt(XUSVt)^+||_2 = ",
    np.linalg.norm(
        np.diag(S) @ Vt @ np.linalg.pinv(X @ U @ np.diag(S) @ Vt), ord=2
    ),
)
print(
    "||S(XUS)^+||_2 = ",
    np.linalg.norm(np.diag(S) @ np.linalg.pinv(X @ U @ np.diag(S)), ord=2),
)
print(
    "||S||_2||(XU)^+||_2 = ",
    np.linalg.norm(np.linalg.pinv(X @ U), ord=2) * S[0],
)

# %%
N = 100
m = 20
r = 10
l = 5
Ymu = np.random.normal(size=(N, r))
Xmu = np.random.normal(size=(N, r + l))
U, S, Vt = np.linalg.svd(Xmu, full_matrices=False)
X = Ymu.T @ U
# A = np.random.normal(size=(N, m))

print(S)
Sigma = np.diag(S)
print(
    "||S(XS)^+||_2^2 = ",
    np.linalg.norm(Sigma @ np.linalg.pinv(X @ Sigma), ord=2) ** 2,
)
print(
    "||(XS)^+||_2^2 = ",
    np.linalg.norm(np.linalg.pinv(X @ Sigma), ord=2) ** 2,  # * S[0]**2,
)
print(
    "||(XS^2X*)^{-1}||_2 = ",
    np.linalg.norm(np.linalg.inv(X @ Sigma**2 @ X.T), ord=2),
)
QX, RX = np.linalg.qr(X.T)
print(QX.shape, RX.shape, Sigma.shape)
print(
    "||S(QS)^+||_2^2 = ",
    np.linalg.norm(Sigma @ np.linalg.pinv(QX.T @ Sigma), ord=2) ** 2,
)
print(
    np.linalg.norm(Sigma @ np.linalg.pinv(QX.T @ Sigma)@np.linalg.pinv(RX.T), ord=2) ** 2,
)

print(
    np.linalg.norm(Sigma**2 @ QX @np.linalg.pinv(QX.T@Sigma**2@QX)@np.linalg.pinv(RX.T), ord=2) ** 2,
)
print(
    np.linalg.norm(Sigma**2 @ QX @np.linalg.pinv(QX.T@Sigma**2@QX),ord=2)**2*np.linalg.norm(np.linalg.pinv(RX.T), ord=2) ** 2,
)
print("--")

print(
    np.linalg.norm(Sigma**2@QX,ord=2)**2*np.linalg.norm(np.linalg.pinv(QX.T@Sigma**2@QX),ord=2)**2*np.linalg.norm(np.linalg.pinv(RX.T), ord=2) ** 2,
)
print(
    np.linalg.norm(Sigma**2@QX,ord=2)**2*np.linalg.norm(QX@np.linalg.pinv(QX.T@Sigma**2@QX)@QX.T,ord=2)**2*np.linalg.norm(np.linalg.pinv(RX.T), ord=2) ** 2,
)
print(
    np.linalg.norm(Sigma**2@QX,ord=2)**2*np.linalg.norm(QX@np.linalg.pinv(QX.T@Sigma**2@QX)@QX.T,ord=2)**2*np.linalg.norm(np.linalg.pinv(RX.T), ord=2) ** 2,
)
print(
    np.linalg.norm(Sigma**2,ord=2)**2*np.linalg.norm(np.linalg.pinv(QX.T@Sigma**2@QX),ord=2)**2*np.linalg.norm(np.linalg.pinv(RX.T), ord=2) ** 2,
)
print(
    np.linalg.norm(Sigma**2,ord=2)**2*np.linalg.norm(np.linalg.pinv(Sigma**2),ord=2)**2*np.linalg.norm(np.linalg.pinv(RX.T), ord=2) ** 2,
)
print(
    np.linalg.cond(Sigma)**4*np.linalg.norm(np.linalg.pinv(X), ord=2) ** 2,
)
print('_')
print(
    np.linalg.norm(
        np.linalg.pinv(X).T
        @ np.linalg.pinv(Sigma) ** 2
        @ QX
        @ QX.T
        @ Sigma**4
        @ QX
        @ QX.T
        @ np.linalg.pinv(Sigma**2)
        @ np.linalg.pinv(X),
        ord=2,
    )
)
print(
    np.linalg.norm(np.linalg.pinv(X), ord=2) ** 2 * np.linalg.cond(Sigma) ** 4
)

test_mat = X @ R
np.linalg.norm(test_mat, ord=2) ** 2, np.linalg.norm(
    test_mat @ test_mat.T, ord=2
), np.linalg.norm(test_mat.T @ test_mat, ord=2)


def pinv(A):
    if A.shape[1] >= A.shape[0]:
        return A.T @ np.linalg.inv(A @ A.T)
    else:
        return pinv(A.T).T


np.linalg.norm(pinv(X @ Sigma).T @ Sigma**2 @ pinv(X @ Sigma), ord=2),
np.linalg.norm(
    np.linalg.inv(X @ Sigma**2 @ X.T) @ X @ Sigma**3 @ pinv(X @ Sigma),
    ord=2,
)
# %%
np.linalg.norm(np.linalg.pinv(RX.T), ord=2), np.linalg.norm(np.linalg.pinv(X), ord=2)
# %%
np.linalg.norm(np.linalg.pinv(QX.T @ Sigma**2 @ QX),ord=2) < np.linalg.norm(np.linalg.pinv(Sigma**2),ord=2)
# %%
"""Let's try to bound the oblique projectors two norm directly."""
from tt_sketch.utils import projector

N = 100
m = 20
r = 10
l = 5
Y = np.random.normal(size=(N, r))
X = np.random.normal(size=(N, r + l))
A = np.random.normal(size=(N, m))

Q, _ = np.linalg.qr(Y)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
Sigma = np.diag(S)
print(np.linalg.norm(projector(X, Y), ord=2))
print(np.linalg.norm(X @ np.linalg.pinv(Y.T @ X) @ Y.T, ord=2))
print(np.linalg.norm(X @ np.linalg.pinv(Y.T @ X), ord=2) * r)
print(Sigma.shape, Y.shape, U.shape, Vt.shape)
print(np.linalg.norm(Sigma @ np.linalg.pinv(Q.T @ U @ Sigma), ord=2))
conjecture = np.linalg.cond(X) ** 2
print(conjecture)

# %%
(
    np.linalg.pinv(Q.T @ U @ Sigma**2 @ U.T @ Q)
    - Q.T @ np.linalg.pinv(U @ Sigma**2 @ U.T) @ Q
)
#%%
np.linalg.norm(np.linalg.pinv(X), ord=2), np.linalg.norm(
    Sigma @ np.linalg.pinv(X @ Sigma), ord=2
)
# %%
norms1 = []
norms2 = []
r = 20
l = 3
sigmas = np.logspace(-3, 0, 20)
for sigma in sigmas:
    norms1_this_sigma = []
    norms2_this_sigma = []

    for i in range(50):
        np.random.seed(i + 123123)
        X = np.random.normal(size=(r, r + l))
        Sigma2 /= Sigma2[0, 0]
        Sigma2 = np.concatenate([np.ones(r), np.zeros(l)])
        Sigma2[r:] = sigma
        Sigma2 = np.diag(Sigma2)

        norm1 = np.linalg.norm(Sigma2 @ np.linalg.pinv(X @ Sigma2), ord=2)
        norms1_this_sigma.append(norm1)

        norm2 = np.linalg.norm(np.linalg.pinv(X), ord=2) * np.linalg.cond(
            Sigma2
        )
        norms2_this_sigma.append(norm2)
    norms1.append(np.mean(norms1_this_sigma))
    norms2.append(np.mean(norms2_this_sigma))
plt.plot(sigmas, norms1, label="||S(XS)^+||_2")
plt.plot(sigmas, norms2, label="||X^+||_2 k(S)")
plt.legend()
plt.xscale("log")
plt.yscale("log")
norms1 < norms2
# %%
r = 10
l = 10

R = np.random.normal(size=(r + l, r + l))
# R = np.triu(R)
_, R = np.linalg.qr(R)
X = np.random.normal(size=(r, r + l))
U, S, Vt = np.linalg.svd(R)
S /= S[0]
Sigma = np.diag(S**10)
cond_r = S[0] / S[-1]

np.linalg.norm(Sigma @ np.linalg.pinv(X @ Sigma), ord=2), np.linalg.norm(
    np.linalg.pinv(X), ord=2
) * cond_r, 1 / np.linalg.norm(X, ord=2)
