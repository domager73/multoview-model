"""Repulsion (diversity) penalty against component merging.

When many bodies come close, the moment criterion has near-degenerate directions
and components collapse.  Add
   P_div = gamma_div * sum_j sum_{m<m'} <u_jm, u_jm'>^2
which in theta-space equals sum (theta_jm^T G^{-1} theta_jm')^2.
Test on r=3 crossing bodies with the grid density.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linear_sum_assignment
import mvlib_web as m

SQRT2PI = np.sqrt(2 * np.pi)


def make_cross(B, r=3, T=14, sigma=0.06, seed=0):
    p0 = np.array([[0.2, 0.5], [0.8, 0.5], [0.5, 0.2]])[:r]
    p1 = np.array([[0.7, 0.45], [0.3, 0.55], [0.5, 0.8]])[:r]
    lam = np.full(r, 1.0 / r)
    F = np.zeros((T, B.N, B.N)); gt = np.zeros((T, r, 2))
    g = lambda v, c: np.exp(-(v - c) ** 2 / (2 * sigma ** 2)) / (sigma * SQRT2PI)
    for t in range(T):
        a = t / (T - 1)
        for k in range(r):
            c = (1 - a) * p0[k] + a * p1[k]; gt[t, k] = c
            F[t] += lam[k] * np.outer(g(B.xg, c[0]), g(B.xg, c[1]))
        F[t] /= F[t].sum() * B.dx * B.dx
    return F, gt


def fit(F, B, r, gamma=1e-4, gamma_div=0.0, restarts=6, seed=0):
    K = B.K
    Zl = [(B.Psi * (F.sum(1) * B.dx)[:, None]).sum(0) * B.dx,
          (B.Psi * (F.sum(0) * B.dx)[:, None]).sum(0) * B.dx]
    Zp = (B.Psi.T @ F @ B.Psi) * B.dx * B.dx
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    S = np.zeros((K - 2, K))
    for i in range(K - 2):
        S[i, i], S[i, i + 1], S[i, i + 2] = 1.0, -2.0, 1.0

    def fg(p):
        th = p[:2 * r * K].reshape(2, r, K); lam = p[2 * r * K:2 * r * K + r]
        Gl = [lam @ th[j] for j in range(2)]; Rl = [Gl[j] - Zl[j] for j in range(2)]
        Rp = np.einsum("m,mk,ml->kl", lam, th[0], th[1]) - Zp
        val = 0.5 * sum(float(a @ a) for a in Rl) + 0.5 * float((Rp ** 2).sum())
        gth = np.zeros((2, r, K)); glm = np.zeros(r)
        if gamma > 0:
            val += 0.5 * gamma * sum(float(np.sum((S @ th[j].T) ** 2)) for j in range(2))
        for j in range(2):
            gth[j] += lam[:, None] * Rl[j][None, :]; glm += th[j] @ Rl[j]
            if gamma > 0:
                gth[j] += gamma * (S.T @ (S @ th[j].T)).T
        gth[0] += lam[:, None] * (th[1] @ Rp.T); gth[1] += lam[:, None] * (th[0] @ Rp)
        glm += np.einsum("mk,kl,ml->m", th[0], Rp, th[1])
        if gamma_div > 0:
            for j in range(2):
                for a in range(r):
                    for b in range(a + 1, r):
                        sim = th[j, a] @ Ginv @ th[j, b]
                        val += 0.5 * gamma_div * sim ** 2
                        gth[j, a] += 2.0 * gamma_div * sim * (Ginv @ th[j, b])
                        gth[j, b] += 2.0 * gamma_div * sim * (Ginv @ th[j, a])
        return val, np.concatenate([gth.ravel(), glm])

    ts = 2 * r * K; ub = np.tile(1.0 / B.C_k, 2 * r)
    bnds = [(0.0, ub[i]) for i in range(ts)] + [(0.0, 1.0)] * r
    cons = [{"type": "eq", "fun": lambda p: p[ts:ts + r].sum() - 1.0,
             "jac": lambda p: np.r_[np.zeros(ts), np.ones(r)]}]
    rg = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        p0 = np.r_[np.clip(np.abs(rg.standard_normal(ts)) * 0.4 + 0.2, 0, ub), np.full(r, 1.0 / r)]
        res = minimize(fg, p0, jac=True, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 3000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x[:ts].reshape(2, r, K); lam = np.clip(best.x[ts:ts + r], 1e-12, None); lam /= lam.sum()
    coef = np.einsum("ik,jmk->jmi", Ginv, th)
    u = np.einsum("jmk,nk->jmn", coef, B.Psi); u = np.maximum(u, 0)
    u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    ctr = np.stack([(u[j] * B.xg[None, :]).sum(1) * B.dx for j in range(2)], 1)
    return ctr


def track(B, F, gt, r, gamma_div):
    rec = np.zeros_like(gt)
    for t in range(len(F)):
        c = fit(F[t], B, r, gamma_div=gamma_div, seed=t)
        if t > 0:
            _, col = linear_sum_assignment(((rec[t - 1][:, None, :] - c[None, :, :]) ** 2).sum(-1)); c = c[col]
        rec[t] = c
    e = []
    for t in range(len(F)):
        C = ((gt[t][:, None, :] - rec[t][None, :, :]) ** 2).sum(-1)
        rs, cs = linear_sum_assignment(C); e.append(np.sqrt(C[rs, cs]))
    return np.concatenate(e).mean(), rec


if __name__ == "__main__":
    B = m.Basis(K=14, N=64)
    F, gt = make_cross(B, r=3, T=14, sigma=0.06, seed=0)
    print("r=3 crossing bodies — centre RMSE:")
    for gd in [0.0, 0.5, 2.0, 8.0]:
        e, rec = track(B, F, gt, 3, gd)
        print(f"  gamma_div={gd:4.1f}: RMSE={e:.4f}")
    e0, rec0 = track(B, F, gt, 3, 0.0)
    eg, recg = track(B, F, gt, 3, 2.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for k in range(3):
        ax[0].plot(gt[:, k, 0], gt[:, k, 1], "k-o", ms=3); ax[0].plot(rec0[:, k, 0], rec0[:, k, 1], "--s", ms=3)
        ax[1].plot(gt[:, k, 0], gt[:, k, 1], "k-o", ms=3); ax[1].plot(recg[:, k, 0], recg[:, k, 1], "--^", ms=3)
    ax[0].set_title("без repulsion (коллапс)"); ax[1].set_title("с repulsion")
    plt.tight_layout(); plt.savefig("repulsion_result.png", dpi=130)
    print("saved repulsion_result.png")
