"""Run the multiview product-mixture model on REAL data.

1) Old Faithful geyser (2D, 272 points) -> kernel density on a grid, fit r products.
2) A MNIST digit image treated as a 2D density, fit r products.
Reports reconstruction error and writes figures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import mvlib_web as m

D = "/var/folders/5b/ybrxv5q135s3z4n4538x1kpw0000gn/T/opencode/data"


def recon(fit, B):
    ctr, lam, th, x = fit
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    Fh = np.einsum("m,mi,mj->ij", lam, u[0], u[1])
    return Fh, u, lam


# ---------- 1) Old Faithful ----------
raw = np.loadtxt(f"{D}/faithful.csv", delimiter=",", skiprows=1, usecols=(1, 2))
X = (raw - raw.min(0)) / (raw.max(0) - raw.min(0))
B = m.Basis(K=12, N=64)
kde = gaussian_kde(X.T, bw_method=0.15)
gx = np.linspace(0, 1, B.N)
GX, GY = np.meshgrid(gx, gx, indexing="ij")
F = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(B.N, B.N)
F /= F.sum() * B.dx * B.dx
print("Old Faithful: points", len(X))
for r in (1, 2, 3):
    fit = m._grid_fit(F, B, r, gamma=1e-4, restarts=6, seed=0)
    Fh, u, lam = recon(fit, B)
    l2 = np.sqrt(np.sum((F - Fh) ** 2) * B.dx * B.dx)
    print(f"  r={r}: L2={l2:.4f}  centres(x,y)={np.round(fit[0],3).tolist()}  lam={np.round(lam,3).tolist()}")

fit2 = m._grid_fit(F, B, 2, gamma=1e-4, restarts=6, seed=0)
Fh2, u2, lam2 = recon(fit2, B)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].scatter(X[:, 0], X[:, 1], s=8, alpha=.5)
ax[0].set_title("Old Faithful: данные (норм.)"); ax[0].set_xlabel("eruptions"); ax[0].set_ylabel("waiting")
ax[1].imshow(F.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma"); ax[1].set_title("плотность (KDE)")
im = ax[2].imshow(Fh2.T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
for c in fit2[0]:
    ax[2].plot(c[0], c[1], "c+", ms=12, mew=2)
ax[2].set_title("наша модель, r=2 (cyan = компоненты)")
plt.tight_layout(); plt.savefig("real_faithful.png", dpi=130)
print("saved real_faithful.png")

# ---------- 2) MNIST digit ----------
import gzip, struct
with gzip.open(f"{D}/train-images.gz", "rb") as f:
    f.read(16)
    n = 8
    imgs = np.frombuffer(f.read(28 * 28 * n), dtype=np.uint8).reshape(n, 28, 28)
# pick a digit that is roughly a blob/curve: use first image (likely 5/0/4/1...)
dig = imgs[0].astype(float)
Bm = m.Basis(K=12, N=28)
Fm = dig / dig.sum()
# resample is not needed: grid is 28x28 == B.N
res = []
for r in (1, 2, 3, 4):
    fit = m._grid_fit(Fm, Bm, r, gamma=1e-4, restarts=5, seed=0)
    Fh, u, lam = recon(fit, Bm)
    l2 = np.sqrt(np.sum((Fm - Fh) ** 2) * Bm.dx * Bm.dx)
    res.append((r, l2, fit, Fh))
    print(f"  MNIST r={r}: L2={l2:.4f}")
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(Fm.T, origin="lower", cmap="gray_r"); ax[0].set_title("MNIST цифра (плотность)")
ax[1].imshow(res[0][3].T, origin="lower", cmap="magma"); ax[1].set_title(f"r=1  L2={res[0][1]:.3f}")
ax[2].imshow(res[1][3].T, origin="lower", cmap="magma"); ax[2].set_title(f"r=2  L2={res[1][1]:.3f}")
ax[3].imshow(res[3][3].T, origin="lower", cmap="magma"); ax[3].set_title(f"r=4  L2={res[3][1]:.3f}")
plt.tight_layout(); plt.savefig("real_mnist.png", dpi=130)
print("saved real_mnist.png")
