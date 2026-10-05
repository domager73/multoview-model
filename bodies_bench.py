"""Different body shapes (separable), moving: when does our moment method beat EM?

Each body m has marginal shape kind in {gauss, uniform, triangular, bimodal, skew},
product u(x)v(y).  We track centres and reconstruct the density.
Methods: moments(gamma=1e-4), moments+temporal, EM(L=2, its default capacity).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import skewnorm
import mvlib_web as m


def marg(kind, c, w, xg):
    if kind == "gauss":
        g = np.exp(-(xg - c) ** 2 / (2 * (w / 3.0) ** 2))
    elif kind == "uniform":
        g = ((xg > c - w / 2) & (xg < c + w / 2)).astype(float)
    elif kind == "triangular":
        g = np.maximum(0.0, 1 - np.abs(xg - c) / (w / 2))
    elif kind == "bimodal":
        g = np.exp(-(xg - (c - w / 3)) ** 2 / (2 * (w / 6.0) ** 2)) + \
            np.exp(-(xg - (c + w / 3)) ** 2 / (2 * (w / 6.0) ** 2))
    elif kind == "skew":
        g = skewnorm.pdf(xg, 6.0, loc=c - w / 4, scale=w / 3)
    else:
        g = np.exp(-(xg - c) ** 2)
    return g / (g.sum() * (xg[1] - xg[0]))


def make_video(B, kind, r=2, T=16, w=0.14, seed=0):
    p0 = np.array([[0.22, 0.25], [0.75, 0.72]])[:r]
    p1 = np.array([[0.30, 0.72], [0.68, 0.28]])[:r]
    lam = np.full(r, 1.0 / r)
    F = np.zeros((T, B.N, B.N)); gt = np.zeros((T, r, 2))
    for t in range(T):
        a = t / (T - 1)
        for k in range(r):
            c = (1 - a) * p0[k] + a * p1[k]
            ux = marg(kind, c[0], w, B.xg); vy = marg(kind, c[1], w, B.xg)
            gtx = (ux * B.xg).sum() * B.dx; gty = (vy * B.xg).sum() * B.dx
            gt[t, k] = [gtx, gty]
            F[t] += lam[k] * np.outer(ux, vy)
        F[t] /= F[t].sum() * B.dx * B.dx
    return F, gt


def centers_moments(F, B, r, seed):
    ctr, lam, th, x = m._grid_fit(F, B, r, gamma=1e-4, restarts=5, seed=seed)
    return ctr


def _centers_from_th(th, B):
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    return np.stack([(u[j] * B.xg[None, :]).sum(1) * B.dx for j in range(2)], 1)


def _sample_X(F, B, seed, n=4000):
    rg = np.random.default_rng(seed); p = F.reshape(-1); p = p / p.sum()
    idx = rg.choice(p.size, size=n, p=p)
    ii, jj = np.unravel_index(idx, F.shape)
    return np.stack([B.xg[ii], B.xg[jj]], 1)


def centers_em(F, B, r, seed, L=2, n=4000):
    X = _sample_X(F, B, seed, n)
    th, lam = m.em_fit(X, B, r, L, 2, n_restarts=4, seed=seed)
    return _centers_from_th(th, B)


def centers_hybrid(F, B, r, seed, L=2, n=4000):
    X = _sample_X(F, B, seed, n)
    th, lam = m.em_from_moments(X, B, r, L, 2, gamma=1e-4, n_restarts=4, seed=seed)
    return _centers_from_th(th, B)


def track(fn, F, gt, r):
    rec = np.zeros_like(gt)
    for t in range(len(F)):
        c = fn(seed=t)
        if t > 0:
            _, col = linear_sum_assignment(((rec[t - 1][:, None, :] - c[None, :, :]) ** 2).sum(-1)); c = c[col]
        rec[t] = c
    return rec


def rms(rec, gt):
    e = []
    for t in range(len(gt)):
        C = ((gt[t][:, None, :] - rec[t][None, :, :]) ** 2).sum(-1)
        rs, cs = linear_sum_assignment(C); e.append(np.sqrt(C[rs, cs]))
    return np.concatenate(e).mean()


def recon_from_th(th, lam, B):
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    return np.einsum("m,mi,mj->ij", lam, u[0], u[1])


def density_and_centers(F, B, r, seed, method):
    if method == "moments":
        ctr, lam, th, x = m._grid_fit(F, B, r, gamma=1e-4, restarts=5, seed=seed)
        return ctr, recon_from_th(th, lam, B)
    X = _sample_X(F, B, seed, 4000)
    if method.startswith("em"):
        L = 2 if method == "em2" else 1
        th, lam = m.em_fit(X, B, r, L, 2, n_restarts=4, seed=seed)
    else:
        th, lam = m.em_from_moments(X, B, r, 2, 2, gamma=1e-4, n_restarts=4, seed=seed)
    return _centers_from_th(th, B), recon_from_th(th, lam, B)


if __name__ == "__main__":
    B = m.Basis(K=12, N=64); METHODS = ["moments", "em1", "em2", "hybrid"]
    print("Центры (RMSE) и форма плотности (отн. ошибка), среднее по кадрам:")
    print(f"{'body':>10} | " + " | ".join(f"{m:>11}" for m in METHODS) + " | winner(sh) | winner(cen)")
    for kind in ["gauss", "uniform", "triangular", "bimodal", "skew"]:
        F, gt = make_video(B, kind, r=2, T=12, seed=1)
        cen = {m: [] for m in METHODS}; de = {m: [] for m in METHODS}
        for t in range(len(F)):
            for meth in METHODS:
                ctr, Fh = density_and_centers(F[t], B, 2, t, meth)
                cen[meth].append(ctr)
                de[meth].append(np.linalg.norm(F[t] - Fh) / np.linalg.norm(F[t]))
        # centres with tracking align
        def track_err(ctrs):
            rec = np.zeros_like(gt)
            for t in range(len(F)):
                c = ctrs[t]
                if t > 0:
                    _, col = linear_sum_assignment(((rec[t-1][:, None, :] - c[None, :, :]) ** 2).sum(-1)); c = c[col]
                rec[t] = c
            return rms(rec, gt)
        ce = {m: track_err(cen[m]) for m in METHODS}
        dea = {m: float(np.mean(de[m])) for m in METHODS}
        ws = min(dea, key=dea.get); wc = min(ce, key=ce.get)
        print(f"{kind:>10} | " + " | ".join(f"{ce[m]:6.3f}/{dea[m]:5.3f}" for m in METHODS) + f" | {ws:>10} | {wc:>11}")
