"""Moving MNIST: track centres of two real MNIST digits moving on a canvas.
Compare our multiview moment method vs EM, against ground truth.
"""
import gzip, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import mvlib_web as m

D = "/var/folders/5b/ybrxv5q135s3z4n4538x1kpw0000gn/T/opencode/data"


def load_mnist(n):
    with gzip.open(f"{D}/train-images.gz", "rb") as f:
        f.read(16)
        return np.frombuffer(f.read(28 * 28 * n), dtype=np.uint8).reshape(n, 28, 28).astype(float)


def make_sequence(imgs, idx, T=20, C=64, seed=0):
    rg = np.random.default_rng(seed)
    frames = np.zeros((T, C, C)); gt = np.zeros((T, 2, 2))
    pos = rg.uniform(4, C - 32, (2, 2)); vel = rg.uniform(1.0, 2.2, (2, 2)) * rg.choice([-1, 1], (2, 2))
    tmpl = []
    for k in range(2):
        im = imgs[idx[k]]
        im = im[im.sum(1) > 0][:, :] if False else im
        ys, xs = np.where(im > 40)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        t = im[y0:y1, x0:x1]
        tmpl.append(t / t.sum())
    for t in range(T):
        for k in range(2):
            pos[k] += vel[k]
            for d in range(2):
                hi = C - tmpl[k].shape[d]
                if pos[k, d] < 0: pos[k, d] = 0; vel[k, d] *= -1
                if pos[k, d] > hi: pos[k, d] = hi; vel[k, d] *= -1
            h, w = tmpl[k].shape
            i0, j0 = int(round(pos[k, 0])), int(round(pos[k, 1]))
            blk = tmpl[k]
            frames[t, i0:i0 + h, j0:j0 + w] += blk
            ys, xs = np.mgrid[0:h, 0:w]
            cy = (blk * ys).sum() + i0; cx = (blk * xs).sum() + j0
            gt[t, k] = [cx / (C - 1), cy / (C - 1)]
        frames[t] /= frames[t].sum()
    return frames, gt


def centers_moments(F, B, r=2, gamma=1e-4, seed=0):
    ctr, lam, th, x = m._grid_fit(F, B, r, gamma=gamma, restarts=5, seed=seed)
    return ctr


def centers_em(F, B, r=2, L=2, seed=0, n=4000):
    rg = np.random.default_rng(seed)
    p = F.reshape(-1); p = p / p.sum()
    idx = rg.choice(p.size, size=n, p=p)
    ii, jj = np.unravel_index(idx, F.shape)
    X = np.stack([B.xg[ii], B.xg[jj]], 1)
    th, lam = m.em_fit(X, B, r, L, 2, n_restarts=3, seed=seed)
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    cx = (u[0] * B.xg[None, :]).sum(1) * B.dx
    cy = (u[1] * B.xg[None, :]).sum(1) * B.dx
    return np.stack([cx, cy], 1)


def err(rec, gt):
    e = []
    for t in range(rec.shape[0]):
        C = ((gt[t][:, None, :] - rec[t][None, :, :]) ** 2).sum(-1)
        rs, cs = linear_sum_assignment(C); e.append(np.sqrt(C[rs, cs]))
    return np.concatenate(e)


if __name__ == "__main__":
    imgs = load_mnist(2000)
    B = m.Basis(K=12, N=64)
    # pick two distinct digits
    idx = [np.where(imgs.reshape(len(imgs), -1).argmax(1) >= 0)[0][0], 1]
    idx = [3, 7]
    F, gt = make_sequence(imgs, idx, T=20, C=64, seed=1)
    rec_m = np.zeros((len(F), 2, 2)); rec_e = np.zeros_like(rec_m); rec_t = rec_m.copy()
    for t in range(len(F)):
        cm = centers_moments(F[t], B, seed=t)
        ce = centers_em(F[t], B, seed=t)
        # align to previous for tracking order
        if t > 0:
            _, cmc = linear_sum_assignment(((rec_m[t-1][:, None, :] - cm[None, :, :]) ** 2).sum(-1)); cm = cm[cmc]
            _, cec = linear_sum_assignment(((rec_e[t-1][:, None, :] - ce[None, :, :]) ** 2).sum(-1)); ce = ce[cec]
        rec_m[t] = cm; rec_e[t] = ce
    # temporal repair for moments
    rec_t[:] = rec_m
    for k in range(2):
        for t in range(1, len(F) - 1):
            pred = 0.5 * (rec_m[t-1, k] + rec_m[t+1, k])
            if np.linalg.norm(rec_m[t, k] - pred) > 0.06:
                rec_t[t, k] = pred
    em_ = err(rec_m, gt); ee = err(rec_e, gt); et = err(rec_t, gt)
    print(f"Moving MNIST (2 real digits, T={len(F)}, 64x64):")
    print(f"  moments          RMSE = {em_.mean():.4f}  ({em_.mean()*63:.1f} px)")
    print(f"  moments+temporal RMSE = {et.mean():.4f}  ({et.mean()*63:.1f} px)")
    print(f"  EM               RMSE = {ee.mean():.4f}  ({ee.mean()*63:.1f} px)")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(F[0], origin="lower", cmap="gray_r"); ax[0].set_title("кадр 1 (реальные цифры)")
    ax[1].imshow(F[10], origin="lower", cmap="gray_r"); ax[1].set_title("кадр 11")
    for k in range(2):
        ax[2].plot(gt[:, k, 0], gt[:, k, 1], "-o", ms=3, c="k", label="true")
        ax[2].plot(rec_m[:, k, 0], rec_m[:, k, 1], "--s", ms=3, label="moments")
        ax[2].plot(rec_t[:, k, 0], rec_t[:, k, 1], "-^", ms=3, label="moments+temporal")
    ax[2].set_title("траектории центров"); ax[2].legend(fontsize=7); ax[2].set_xlabel("x"); ax[2].set_ylabel("y")
    ax[3].bar(["moments", "+temporal", "EM"], [em_.mean(), et.mean(), ee.mean()], color=["#6ea8fe", "#3ddc97", "#ffb454"])
    ax[3].set_title("RMSE трекинга центров (норм.)")
    plt.tight_layout(); plt.savefig("moving_mnist_result.png", dpi=130)
    print("saved moving_mnist_result.png")
