"""Test 1: separable moving bodies (rectangles) -> fair benchmark, GT centres.
Test 2: non-separable shapes -> extend a component to a SUM of products (rank R),
        compare with the optimal (SVD) rank-R product approximation.
"""
import gzip, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import mvlib_web as m

D = "/var/folders/5b/ybrxv5q135s3z4n4538x1kpw0000gn/T/opencode/data"


# ============================ TEST 1 ============================
def make_rect_video(B, r=2, T=16, seed=0):
    rg = np.random.default_rng(seed)
    p0 = np.array([[0.22, 0.25], [0.75, 0.72]])[:r]
    p1 = np.array([[0.30, 0.72], [0.68, 0.28]])[:r]
    wh = rg.uniform(0.10, 0.16, (r, 2))
    lam = np.full(r, 1.0 / r)
    F = np.zeros((T, B.N, B.N)); gt = np.zeros((T, r, 2))
    for t in range(T):
        a = t / (T - 1)
        for k in range(r):
            c = (1 - a) * p0[k] + a * p1[k]; gt[t, k] = c
            ux = ((B.xg > c[0] - wh[k, 0] / 2) & (B.xg < c[0] + wh[k, 0] / 2)).astype(float)
            vy = ((B.xg > c[1] - wh[k, 1] / 2) & (B.xg < c[1] + wh[k, 1] / 2)).astype(float)
            ux /= ux.sum() * B.dx; vy /= vy.sum() * B.dx
            F[t] += lam[k] * np.outer(ux, vy)
        F[t] /= F[t].sum() * B.dx * B.dx
    return F, gt


def centers_moments(F, B, r, seed):
    ctr, lam, th, x = m._grid_fit(F, B, r, gamma=1e-4, restarts=5, seed=seed)
    return ctr


def centers_em(F, B, r, L=3, seed=0, n=4000):
    rg = np.random.default_rng(seed); p = F.reshape(-1); p = p / p.sum()
    idx = rg.choice(p.size, size=n, p=p)
    ii, jj = np.unravel_index(idx, F.shape)
    X = np.stack([B.xg[ii], B.xg[jj]], 1)
    th, lam = m.em_fit(X, B, r, L, 2, n_restarts=3, seed=seed)
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    return np.stack([(u[j] * B.xg[None, :]).sum(1) * B.dx for j in range(2)], 1)


def track(centers_fn, F, gt, r):
    T = len(F); rec = np.zeros((T, r, 2))
    for t in range(T):
        c = centers_fn(F[t], seed=t)
        if t > 0:
            _, col = linear_sum_assignment(((rec[t - 1][:, None, :] - c[None, :, :]) ** 2).sum(-1)); c = c[col]
        rec[t] = c
    return rec


def err(rec, gt):
    e = []
    for t in range(len(gt)):
        C = ((gt[t][:, None, :] - rec[t][None, :, :]) ** 2).sum(-1)
        rs, cs = linear_sum_assignment(C); e.append(np.sqrt(C[rs, cs]))
    return np.concatenate(e)


# ============================ TEST 2 ============================
def load_mnist(n):
    with gzip.open(f"{D}/train-images.gz", "rb") as f:
        f.read(16)
        return np.frombuffer(f.read(28 * 28 * n), dtype=np.uint8).reshape(n, 28, 28).astype(float)


if __name__ == "__main__":
    # ---- Test 1 ----
    B = m.Basis(K=12, N=64)
    F, gt = make_rect_video(B, r=2, T=16, seed=2)
    rec_m = track(lambda Ft, seed: centers_moments(Ft, B, 2, seed), F, gt, 2)
    rec_e = track(lambda Ft, seed: centers_em(Ft, B, 2, seed=seed), F, gt, 2)
    rec_t = rec_m.copy()
    for k in range(2):
        for t in range(1, len(F) - 1):
            pred = 0.5 * (rec_m[t - 1, k] + rec_m[t + 1, k])
            if np.linalg.norm(rec_m[t, k] - pred) > 0.05: rec_t[t, k] = pred
    em_, ee, et = err(rec_m, gt), err(rec_e, gt), err(rec_t, gt)
    print("TEST 1 — сепарабельные тела (прямоугольники), GT известен:")
    print(f"  моменты          RMSE={em_.mean():.4f} ({em_.mean()*63:.1f}px)")
    print(f"  моменты+temporal RMSE={et.mean():.4f} ({et.mean()*63:.1f}px)")
    print(f"  EM               RMSE={ee.mean():.4f} ({ee.mean()*63:.1f}px)")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(F[0], origin="lower", cmap="magma"); ax[0].set_title("кадр: сепарабельные тела")
    for k in range(2):
        ax[1].plot(gt[:, k, 0], gt[:, k, 1], "k-o", ms=3)
        ax[1].plot(rec_m[:, k, 0], rec_m[:, k, 1], "--s", ms=3, label="moments")
        ax[1].plot(rec_e[:, k, 0], rec_e[:, k, 1], ":^", ms=3, label="EM")
    ax[1].set_title("траектории: чёрное=GT"); ax[1].legend(fontsize=7)
    ax[2].bar(["moments", "+temp", "EM"], [em_.mean(), et.mean(), ee.mean()], color=["#6ea8fe", "#3ddc97", "#ffb454"])
    ax[2].set_title("RMSE центров (норм.)")
    plt.tight_layout(); plt.savefig("sep_bodies_result.png", dpi=130)
    print("saved sep_bodies_result.png")

    # ---- Test 2 ----
    imgs = load_mnist(6); dig = imgs[0].astype(float)
    Bm = m.Basis(K=16, N=28); dx = Bm.dx
    Fd = dig / (dig.sum() * dx * dx)          # proper density: int F dx^2 = 1
    def relerr(A):
        return float(np.linalg.norm(Fd - A) / np.linalg.norm(Fd))
    U, S, Vh = np.linalg.svd(Fd)
    print("\nTEST 2 — реальная форма (цифра) как СУММА произведений (отн. ошибка):")
    for R in [1, 2, 3, 5, 8, 12, 16]:
        approx = np.maximum((U[:, :R] * S[:R]) @ Vh[:R], 0)
        print(f"  SVD rank R={R:2d}: rel={relerr(approx):.4f}")
    for R in [1, 2, 3]:
        ctr, lam, th, x = m._grid_fit(Fd, Bm, R, gamma=1e-4, restarts=5, seed=0)
        G = (Bm.Psi.T @ Bm.Psi) * dx; Ginv = np.linalg.pinv(G)
        u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), Bm.Psi)
        u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * dx, 1e-30)
        Fh = np.einsum("m,mi,mj->ij", lam, u[0], u[1])
        print(f"  our nonneg model r=R={R}: rel={relerr(Fh):.4f}")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(Fd.T, origin="lower", cmap="gray_r"); ax[0].set_title("цифра (плотность)")
    for i, R in enumerate([2, 5, 12]):
        approx = np.maximum((U[:, :R] * S[:R]) @ Vh[:R], 0)
        ax[i + 1].imshow(approx.T, origin="lower", cmap="magma"); ax[i + 1].set_title(f"SVD rank {R}")
    plt.tight_layout(); plt.savefig("rank_products_result.png", dpi=130)
    print("saved rank_products_result.png")
