"""Browser-side core (runs under Pyodide). Reuses the same method as the notebooks.

run_pipeline(cfg_json) -> json string with metrics + arrays for plotting.
"""
import json
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

SQRT2PI = np.sqrt(2.0 * np.pi)


class Basis:
    def __init__(self, K=10, N=801, smax=1.0, ratio=np.sqrt(2.0)):
        self.K, self.N = K, N
        self.xg = np.linspace(0.0, 1.0, N)
        self.dx = self.xg[1] - self.xg[0]
        self.centers = np.linspace(0.0, 1.0, K)
        self.sigmas = smax * ratio ** (np.arange(1, K + 1) - K)
        Pu = np.exp(-(self.xg[:, None] - self.centers[None, :]) ** 2
                    / (2.0 * self.sigmas[None, :] ** 2))
        self.C_k = Pu.sum(0) * self.dx
        self.Psi = Pu / self.C_k


def make_u_mix(B, kind, d, r, L, seed=0):
    xg, dx, rg = B.xg, B.dx, np.random.default_rng(seed)
    u = np.zeros((d, r, B.N))
    for j in range(d):
        for m in range(r):
            w = rg.uniform(0.5, 1.5, L); w /= w.sum()
            if kind == "gauss":
                xi = np.sort(rg.uniform(0.15, 0.85, L)) if L > 1 else rg.uniform(0.3, 0.7, 1)
                for i in range(1, L):
                    if xi[i] - xi[i - 1] < 0.12:
                        xi[i] = min(0.9, xi[i - 1] + 0.12)
                sg = rg.uniform(0.03, 0.09, L)
                raw = sum(w[l] * np.exp(-(xg - xi[l]) ** 2 / (2 * sg[l] ** 2))
                          / (SQRT2PI * sg[l]) for l in range(L))
            elif kind == "beta":
                raw = np.zeros_like(xg)
                for l in range(L):
                    a, b = (rg.uniform(1.5, 3), rg.uniform(4, 9)) if l % 2 == 0 \
                        else (rg.uniform(4, 9), rg.uniform(1.5, 3))
                    g = xg ** (a - 1) * (1 - xg) ** (b - 1)
                    raw += w[l] * g / (g.sum() * dx)
            elif kind == "heavy":
                xi = np.sort(rg.uniform(0.2, 0.8, L)) if L > 1 else rg.uniform(0.3, 0.7, 1)
                sc = rg.uniform(0.03, 0.07, L); nu = 2.0
                raw = sum(w[l] * (1 + ((xg - xi[l]) / sc[l]) ** 2 / nu) ** (-(nu + 1) / 2) / sc[l]
                          for l in range(L))
            elif kind == "uniform":
                raw = np.zeros_like(xg)
                for l in range(L):
                    c = rg.uniform(0.2, 0.7); wd = rg.uniform(0.08, 0.2)
                    raw += w[l] * ((xg > c) & (xg < c + wd)) / wd
            elif kind == "laplace":
                xi = rg.uniform(0.2, 0.8, L); sc = rg.uniform(0.03, 0.07, L)
                raw = sum(w[l] * np.exp(-np.abs(xg - xi[l]) / sc[l]) / (2 * sc[l])
                          for l in range(L))
            elif kind == "skew":
                # asymmetric skew-normal kernels, alternating direction of skew
                from scipy.stats import skewnorm
                xi = np.sort(rg.uniform(0.2, 0.8, L)) if L > 1 else rg.uniform(0.3, 0.7, 1)
                sc = rg.uniform(0.05, 0.12, L)
                al = rg.uniform(3.0, 8.0, L) * np.where(np.arange(L) % 2 == 0, 1.0, -1.0)
                raw = sum(w[l] * skewnorm.pdf(xg, al[l], loc=xi[l], scale=sc[l])
                          for l in range(L))
            else:
                raw = np.ones_like(xg)
            u[j, m] = raw / (raw.sum() * dx)
    lam = rg.uniform(0.5, 1.5, r); lam /= lam.sum()
    return u, lam


def sample(u, lam, B, n, seed=0, contam=0.0):
    rg = np.random.default_rng(seed)
    d, r = u.shape[0], u.shape[1]
    cum = np.cumsum(u, axis=2) / np.cumsum(u, axis=2)[:, :, -1:]
    idx = np.zeros((n, d), dtype=int)
    sel = rg.choice(r, n, p=lam)
    rr = rg.random((n, d))
    for j in range(d):
        idx[:, j] = (cum[j, sel] < rr[:, j:j + 1]).sum(1)
    idx = np.clip(idx, 0, B.N - 1)
    X = B.xg[idx]
    if contam > 0:
        bad = rg.random(n) < contam
        X[bad] = rg.random((bad.sum(), d))
    return idx, X


def theta_true_of(u, B):
    return np.einsum("jmn,nk->jmk", u, B.Psi) * B.dx


def _loss_grad(p, Zl, Zp, d, r, K, pairs, gamma, S, grad):
    th = p[:d * r * K].reshape(d, r, K)
    lam = p[d * r * K:d * r * K + r]
    Gl = [lam @ th[j] for j in range(d)]
    Gp = {pr: np.einsum("m,mk,ml->kl", lam, th[pr[0]], th[pr[1]]) for pr in pairs}
    Rl = [Gl[j] - Zl[j] for j in range(d)]
    Rp = {pr: Gp[pr] - Zp[pr] for pr in pairs}
    val = 0.5 * sum(float(a @ a) for a in Rl) + 0.5 * sum(float((Rp[pr] ** 2).sum()) for pr in pairs)
    if gamma > 0:
        val += 0.5 * gamma * sum(float(np.sum((S @ th[j].T) ** 2)) for j in range(d))
    if not grad:
        return val
    gth = np.zeros((d, r, K)); glm = np.zeros(r)
    for j in range(d):
        gth[j] += lam[:, None] * Rl[j][None, :]; glm += th[j] @ Rl[j]
        if gamma > 0:
            gth[j] += gamma * (S.T @ (S @ th[j].T)).T
    for pr in pairs:
        R = Rp[pr]
        gth[pr[0]] += lam[:, None] * (th[pr[1]] @ R.T)
        gth[pr[1]] += lam[:, None] * (th[pr[0]] @ R)
        glm += np.einsum("mk,kl,ml->m", th[pr[0]], R, th[pr[1]])
    return val, np.concatenate([gth.ravel(), glm])


def moments_theta(X, B, r, d, gamma, n_restarts, seed):
    K = B.K
    pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
    P = [np.exp(-(X[:, j][:, None] - B.centers[None, :]) ** 2
               / (2 * B.sigmas[None, :] ** 2)) / B.C_k[None, :] for j in range(d)]
    Zl = [P[j].mean(0) for j in range(d)]
    Zp = {pr: (P[pr[0]].T @ P[pr[1]]) / X.shape[0] for pr in pairs}
    ts = d * r * K
    S = np.zeros((K - 2, K))
    for i in range(K - 2):
        S[i, i], S[i, i + 1], S[i, i + 2] = 1.0, -2.0, 1.0
    ub = np.tile(1.0 / B.C_k, d * r)
    bnds = [(0.0, ub[i]) for i in range(ts)] + [(0.0, 1.0)] * r
    cons = [{"type": "eq", "fun": lambda p: p[ts:ts + r].sum() - 1.0,
             "jac": lambda p: np.r_[np.zeros(ts), np.ones(r)]}]
    rg = np.random.default_rng(seed); best = None
    for _ in range(n_restarts):
        p0 = np.r_[np.clip(np.abs(rg.standard_normal(ts)) * 0.5 + 0.2, 0, ub), np.full(r, 1.0 / r)]
        res = minimize(_loss_grad, p0, args=(Zl, Zp, d, r, K, pairs, gamma, S, True),
                       jac=True, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 3000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x[:ts].reshape(d, r, K)
    lam = np.clip(best.x[ts:ts + r], 1e-12, None); lam /= lam.sum()
    return th, lam


def em_fit(X, B, r, L, d, n_restarts, seed, n_iter=200, init=None):
    n = X.shape[0]; rg = np.random.default_rng(seed); fl = B.sigmas.min() / 2.0
    best = None
    for it_r in range(n_restarts):
        if init is not None and it_r == 0:
            lam, XI, SG, MU = (np.array(a, float).copy() for a in init)
        else:
            lam = rg.dirichlet(np.ones(r))
        XI = np.array([[np.sort(rg.uniform(0.1, 0.9, L)) for _ in range(r)] for _ in range(d)])
        SG = np.array([[rg.uniform(2 * fl, 0.15, L) for _ in range(r)] for _ in range(d)])
        MU = np.full((d, r, L), 1.0 / L)
        prev = -np.inf
        for it in range(n_iter):
            logu = np.zeros((n, d, r)); comp = np.zeros((n, d, r, L))
            for j in range(d):
                for m in range(r):
                    c = np.stack([MU[j, m, l] * np.exp(-(X[:, j] - XI[j, m, l]) ** 2
                                 / (2 * SG[j, m, l] ** 2)) / (SQRT2PI * SG[j, m, l])
                                  for l in range(L)], axis=1)
                    comp[:, j, m, :] = c; logu[:, j, m] = np.log(c.sum(1) + 1e-300)
            lg = np.log(lam + 1e-300)[None, :] + logu.sum(1)
            mx = lg.max(1, keepdims=True); w = np.exp(lg - mx)
            ll = float((np.log(w.sum(1)) + mx[:, 0]).sum())
            gam = w / w.sum(1, keepdims=True)
            lam = np.clip(gam.mean(0), 1e-8, None); lam /= lam.sum()
            for j in range(d):
                for m in range(r):
                    cs = comp[:, j, m, :].sum(1, keepdims=True) + 1e-300
                    dlt = gam[:, m:m + 1] * comp[:, j, m, :] / cs
                    s = dlt.sum(0) + 1e-12
                    MU[j, m] = s / s.sum()
                    XI[j, m] = (dlt * X[:, j:j + 1]).sum(0) / s
                    v = (dlt * (X[:, j:j + 1] - XI[j, m][None, :]) ** 2).sum(0) / s
                    SG[j, m] = np.sqrt(np.maximum(v, fl ** 2))
            if abs(ll - prev) < 1e-9 * abs(ll):
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, lam.copy(), XI.copy(), SG.copy(), MU.copy())
    _, lam, XI, SG, MU = best
    th = np.zeros((d, r, B.K))
    for j in range(d):
        for m in range(r):
            uh = sum(MU[j, m, l] * np.exp(-(B.xg - XI[j, m, l]) ** 2
                     / (2 * SG[j, m, l] ** 2)) / (SQRT2PI * SG[j, m, l]) for l in range(L))
            th[j, m] = (uh[:, None] * B.Psi).sum(0) * B.dx
    return th, lam


def _fit_component(theta_jm, B, L, n_restarts=6, seed=0):
    from scipy.optimize import nnls
    lo = np.log(B.sigmas.min()); hi = np.log(B.sigmas.max()); s2 = B.sigmas ** 2
    rg = np.random.default_rng(seed)

    def A_mat(xi, sig):
        A = np.zeros((B.K, L))
        for l in range(L):
            D = s2 + sig[l] ** 2
            A[:, l] = (1.0 / B.C_k) * (1.0 + sig[l] ** 2 / s2) ** -0.5 * \
                np.exp(-(xi[l] - B.centers) ** 2 / (2.0 * D))
        return A

    def obj(p):
        xi = p[:L]; sig = np.exp(np.clip(p[L:], lo, hi))
        A = A_mat(xi, sig); mu, _ = nnls(A, theta_jm); r = theta_jm - A @ mu
        return float(r @ r) if np.isfinite(r).all() else 1e10

    bnds = [(0.0, 1.0)] * L + [(lo, hi)] * L
    best = None
    for _ in range(n_restarts):
        p0 = np.r_[np.sort(rg.uniform(0.1, 0.9, L)), rg.uniform(lo, hi, L)]
        res = minimize(obj, p0, method="SLSQP", bounds=bnds, options={"maxiter": 1500})
        if best is None or res.fun < best.fun:
            best = res
    xi = best.x[:L]; sig = np.exp(np.clip(best.x[L:], lo, hi))
    mu, _ = nnls(A_mat(xi, sig), theta_jm)
    o = np.argsort(xi)
    return xi[o], sig[o], mu[o]


def _theta_to_gaussians(theta, lam, B, L, seed=0):
    d, r = theta.shape[0], theta.shape[1]
    XI = np.zeros((d, r, L)); SG = np.zeros((d, r, L)); MU = np.zeros((d, r, L))
    for j in range(d):
        for m in range(r):
            XI[j, m], SG[j, m], MU[j, m] = _fit_component(theta[j, m], B, L, seed=seed + j * r + m)
    return lam.copy(), XI, SG, MU


def em_from_moments(X, B, r, L, d, gamma, n_restarts, seed):
    th, lam = moments_theta(X, B, r, d, gamma, max(2, n_restarts // 2), seed)
    init = _theta_to_gaussians(th, lam, B, L, seed=seed)
    return em_fit(X, B, r, L, d, 2, seed, init=init)


def align(tt, lt, te, le):
    Tt = tt.transpose(1, 0, 2).reshape(tt.shape[1], -1)
    Te = te.transpose(1, 0, 2).reshape(te.shape[1], -1)
    C = ((Tt[:, None, :] - Te[None, :, :]) ** 2).sum(-1) + (lt[:, None] - le[None, :]) ** 2
    _, pm = linear_sum_assignment(C)
    return pm


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


def run_pipeline(cfg_json):
    cfg = json.loads(cfg_json)
    d = int(cfg["d"]); r = int(cfg["r"]); L = int(cfg["L"]); K = int(cfg["K"])
    n = int(cfg["n"]); kind = cfg["kind"]; gamma = float(cfg["gamma"])
    contam = float(cfg.get("contam", 0.0)); seed = int(cfg.get("seed", 0))
    grids = int(cfg.get("grid", 120))
    B = Basis(K=K, N=801)
    u, lam = make_u_mix(B, kind, d, r, L, seed=seed)
    tt = theta_true_of(u, B)
    idx, X = sample(u, lam, B, n, seed=seed + 100, contam=contam)

    out = {"info": {
        "ident_ok": bool(r <= (d - 1) * (K - 1) / 4 + 1e-9),
        "ident_bound": float((d - 1) * (K - 1) / 4),
        "n_moments": int(d * K + K * K * d * (d - 1) // 2),
        "n_params": int(d * r * K + r),
    }, "metrics": {}}

    import time
    results = {}
    t0 = time.time(); th_b, lm_b = moments_theta(X, B, r, d, 0.0, 5, seed); t_b = time.time() - t0
    t0 = time.time(); th_m, lm_m = moments_theta(X, B, r, d, gamma, 5, seed); t_m = time.time() - t0
    t0 = time.time(); th_e, lm_e = em_fit(X, B, r, L, d, 4, seed); t_e = time.time() - t0
    t0 = time.time(); th_h, lm_h = em_from_moments(X, B, r, L, d, gamma, 5, seed); t_h = time.time() - t0
    results["baseline"] = (th_b, lm_b, t_b)
    results["moments"] = (th_m, lm_m, t_m)
    results["em"] = (th_e, lm_e, t_e)
    results["hybrid"] = (th_h, lm_h, t_h)
    for name, (th, lm, t) in results.items():
        pm = align(tt, lam, th, lm)
        out["metrics"][name] = {
            "rel_theta": rel(tt, np.stack([th[j][pm] for j in range(d)])),
            "rel_lam": rel(lam, lm[pm]),
            "time": t,
        }

    th, lm, _ = results[cfg.get("method", "moments")]
    pm = align(tt, lam, th, lm)
    th_a = np.stack([th[j][pm] for j in range(d)])
    lam_a = lm[pm]

    # rebuild u_est from theta_est via the dual (Gram) synthesis: with
    # theta_i = <u, psi_i> and u = sum_k c_k psi_k we need c = G^{-1} theta,
    # because psi_k are L1-normalised (heavily overlapping), not orthonormal.
    G = (B.Psi.T @ B.Psi) * B.dx
    Ginv = np.linalg.pinv(G)
    coef = np.einsum("ik,jmk->jmi", Ginv, th_a)       # (d,r,K)
    u_est = np.einsum("jmk,nk->jmn", coef, B.Psi)      # (d,r,N)
    u_est = np.maximum(u_est, 0.0)
    u_est /= np.maximum(u_est.sum(2, keepdims=True) * B.dx, 1e-30)

    # downsample everything to `grids` points for the browser
    ids = np.clip(np.round(np.linspace(0, 1, grids) * (B.N - 1)).astype(int), 0, B.N - 1)
    gx = (B.xg[ids]).tolist()
    out["plot"] = {
        "x": gx,
        "u_true": [[u[j, m][ids].tolist() for m in range(r)] for j in range(d)],
        "u_est": [[u_est[j, m][ids].tolist() for m in range(r)] for j in range(d)],
        "lam_true": lam.tolist(),
        "lam_est": lam_a.tolist(),
    }
    if d == 2:
        f_true = np.einsum("m,mi,mj->ij", lam, u[0], u[1])
        f_est = np.einsum("m,mi,mj->ij", lam_a, u_est[0], u_est[1])
        out["plot"]["f_true"] = f_true[np.ix_(ids, ids)].tolist()
        out["plot"]["f_est"] = f_est[np.ix_(ids, ids)].tolist()
        out["plot"]["grid2"] = gx
    out["theta"] = {"true": tt.reshape(-1).tolist(), "est": th_a.reshape(-1).tolist()}
    return json.dumps(out)


# --------------------------------------------------------------------------- #
# Video mode: moving bodies recovered frame-by-frame by the multiview model
# --------------------------------------------------------------------------- #
def _make_video(B, r, T, sigma, seed, crossing=True):
    p0 = np.array([[0.22, 0.28], [0.78, 0.72], [0.5, 0.2]])[:r]
    p1 = np.array([[0.30, 0.70], [0.70, 0.28], [0.5, 0.8]])[:r] if crossing else \
         np.array([[0.62, 0.30], [0.38, 0.70], [0.8, 0.5]])[:r]
    lam = np.full(r, 1.0 / r)
    F = np.zeros((T, B.N, B.N))
    ctr = np.zeros((T, r, 2))
    for t in range(T):
        a = t / (T - 1)
        for m in range(r):
            c = (1 - a) * p0[m] + a * p1[m]
            ctr[t, m] = c
            u = np.exp(-(B.xg - c[0]) ** 2 / (2 * sigma ** 2)) / (sigma * SQRT2PI)
            v = np.exp(-(B.xg - c[1]) ** 2 / (2 * sigma ** 2)) / (sigma * SQRT2PI)
            F[t] += lam[m] * np.outer(u, v)
        F[t] /= F[t].sum() * B.dx * B.dx
    return F, ctr, lam


def _grid_fit(F, B, r, gamma=1e-4, restarts=3, seed=0, init=None):
    K = B.K
    Zl = [(B.Psi * (F.sum(1) * B.dx)[:, None]).sum(0) * B.dx,
          (B.Psi * (F.sum(0) * B.dx)[:, None]).sum(0) * B.dx]
    Zp = {(0, 1): (B.Psi.T @ F @ B.Psi) * B.dx * B.dx}
    pairs = [(0, 1)]

    def fg(p):
        th = p[:2 * r * K].reshape(2, r, K); lam = p[2 * r * K:2 * r * K + r]
        Gl = [lam @ th[j] for j in range(2)]
        Gp = np.einsum("m,mk,ml->kl", lam, th[0], th[1])
        Rl = [Gl[j] - Zl[j] for j in range(2)]; Rp = Gp - Zp[(0, 1)]
        S = np.zeros((K - 2, K))
        for i in range(K - 2):
            S[i, i], S[i, i + 1], S[i, i + 2] = 1.0, -2.0, 1.0
        val = 0.5 * sum(float(a @ a) for a in Rl) + 0.5 * float((Rp ** 2).sum())
        if gamma > 0:
            val += 0.5 * gamma * sum(float(np.sum((S @ th[j].T) ** 2)) for j in range(2))
        gth = np.zeros((2, r, K)); glm = np.zeros(r)
        for j in range(2):
            gth[j] += lam[:, None] * Rl[j][None, :]; glm += th[j] @ Rl[j]
            if gamma > 0:
                gth[j] += gamma * (S.T @ (S @ th[j].T)).T
        gth[0] += lam[:, None] * (th[1] @ Rp.T)
        gth[1] += lam[:, None] * (th[0] @ Rp)
        glm += np.einsum("mk,kl,ml->m", th[0], Rp, th[1])
        return val, np.concatenate([gth.ravel(), glm])

    ts = 2 * r * K
    ub = np.tile(1.0 / B.C_k, 2 * r)
    bnds = [(0.0, ub[i]) for i in range(ts)] + [(0.0, 1.0)] * r
    cons = [{"type": "eq", "fun": lambda p: p[ts:ts + r].sum() - 1.0,
             "jac": lambda p: np.r_[np.zeros(ts), np.ones(r)]}]
    rg = np.random.default_rng(seed); best = None
    starts = []
    if init is not None:
        starts.append(init)
    for _ in range(restarts - len(starts)):
        starts.append(np.r_[np.clip(np.abs(rg.standard_normal(ts)) * 0.4 + 0.2, 0, ub), np.full(r, 1.0 / r)])
    for p0 in starts:
        res = minimize(fg, p0, jac=True, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x[:ts].reshape(2, r, K); lam = np.clip(best.x[ts:ts + r], 1e-12, None); lam /= lam.sum()
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    coef = np.einsum("ik,jmk->jmi", Ginv, th)
    u = np.einsum("jmk,nk->jmn", coef, B.Psi); u = np.maximum(u, 0)
    u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    ctr = np.stack([(u[j] * B.xg[None, :]).sum(1) * B.dx for j in range(2)], 1)  # (r,2)
    return ctr, lam, th, best.x

def _centers_from_th(th, B, d):
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    return np.stack([(u[j] * B.xg[None, :]).sum(1) * B.dx for j in range(d)], 1)


def run_video(cfg_json):
    cfg = json.loads(cfg_json)
    r = int(cfg.get("r", 2)); T = int(cfg.get("T", 12))
    sigma = float(cfg.get("sigma", 0.055)); seed = int(cfg.get("seed", 0))
    K = int(cfg.get("K", 16)); N = int(cfg.get("N", 64)); P = int(cfg.get("P", 48))
    crossing = bool(cfg.get("crossing", True)); gamma = float(cfg.get("gamma", 1e-4))
    B = Basis(K=K, N=N)
    F, ctrue, lam = _make_video(B, r, T, sigma, seed, crossing)

    rec_i = np.zeros_like(ctrue); rec_t = np.zeros_like(ctrue)
    for t in range(T):
        ctr, lm, th, x = _grid_fit(F[t], B, r, gamma=gamma, restarts=4, seed=t)
        if t > 0:                      # order components to continue the previous frame
            C = ((rec_i[t - 1][:, None, :] - ctr[None, :, :]) ** 2).sum(-1)
            rows, cols = linear_sum_assignment(C)
            ctr = ctr[cols]
        rec_i[t] = ctr
    # temporal repair: outliers vs linear prediction replaced by interpolation
    thr = 0.5 * sigma
    rec_t = rec_i.copy()
    for m in range(r):
        for t in range(1, T - 1):
            pred = 0.5 * (rec_i[t - 1, m] + rec_i[t + 1, m])
            if np.linalg.norm(rec_i[t, m] - pred) > thr:
                rec_t[t, m] = pred
    # errors vs truth (match columns)
    # optional hybrid (moment-init EM) centres
    rec_h = None
    if bool(cfg.get("hybrid", False)):
        rec_h = np.zeros_like(ctrue)
        for t in range(T):
            rg = np.random.default_rng(t); p = F[t].reshape(-1); p = p / p.sum()
            idx = rg.choice(p.size, size=min(2500, p.size), p=p)
            ii, jj = np.unravel_index(idx, F[t].shape)
            Xh = np.stack([B.xg[ii], B.xg[jj]], 1)
            thh, lmh = em_from_moments(Xh, B, r, 2, 2, gamma=gamma, n_restarts=3, seed=t)
            c = _centers_from_th(thh, B, 2)
            if t > 0:
                C = ((rec_h[t - 1][:, None, :] - c[None, :, :]) ** 2).sum(-1)
                _, col = linear_sum_assignment(C); c = c[col]
            rec_h[t] = c
    def err(a):
        e = []
        for t in range(T):
            C = ((ctrue[t][:, None, :] - a[t][None, :, :]) ** 2).sum(-1)
            rows, cols = linear_sum_assignment(C)
            e.append(np.sqrt(C[rows, cols]).mean())
        return e
    ei, et = err(rec_i), err(rec_t)
    idxs = np.clip(np.round(np.linspace(0, N - 1, P)).astype(int), 0, N - 1)
    out = {
        "r": r, "T": T, "P": P, "sigma": sigma,
        "frames": [F[t][np.ix_(idxs, idxs)].tolist() for t in range(T)],
        "true": ctrue.tolist(),
        "rec_indep": rec_i.tolist(),
        "rec_temp": rec_t.tolist(),
        "rec_hyb": (rec_h.tolist() if rec_h is not None else None),
        "err_indep": ei, "err_temp": et,
        "mean_indep": float(np.mean(ei)), "mean_temp": float(np.mean(et)),
    }
    return json.dumps(out)


# --------------------------------------------------------------------------- #
# 3D video mode: bodies moving in 3D, decomposed frame-by-frame (d=3)
# --------------------------------------------------------------------------- #
def _make_video3d(B, r, T, sigma, seed):
    p0 = np.array([[0.25, 0.25, 0.25], [0.75, 0.70, 0.30], [0.30, 0.75, 0.70]])[:r]
    p1 = np.array([[0.30, 0.70, 0.75], [0.70, 0.30, 0.70], [0.70, 0.30, 0.30]])[:r]
    lam = np.full(r, 1.0 / r)
    N = B.N
    F = np.zeros((T, N, N, N)); ctr = np.zeros((T, r, 3))
    g = lambda v, c: np.exp(-(v - c) ** 2 / (2 * sigma ** 2))
    for t in range(T):
        a = t / (T - 1)
        for m in range(r):
            c = (1 - a) * p0[m] + a * p1[m]; ctr[t, m] = c
            u = g(B.xg, c[0]) / (sigma * SQRT2PI)
            v = g(B.xg, c[1]) / (sigma * SQRT2PI)
            w = g(B.xg, c[2]) / (sigma * SQRT2PI)
            F[t] += lam[m] * np.einsum("a,b,c->abc", u, v, w)
        F[t] /= F[t].sum() * B.dx ** 3
    return F, ctr


def _grid_fit3d(F, B, r, gamma=1e-4, restarts=3, seed=0):
    K = B.K; d = 3
    Zl = [(B.Psi * (F.sum((1, 2)) * B.dx * B.dx)[:, None]).sum(0) * B.dx,
          (B.Psi * (F.sum((0, 2)) * B.dx * B.dx)[:, None]).sum(0) * B.dx,
          (B.Psi * (F.sum((0, 1)) * B.dx * B.dx)[:, None]).sum(0) * B.dx]
    Zp = {(0, 1): (B.Psi.T @ F.sum(2) @ B.Psi) * B.dx ** 3,
          (0, 2): (B.Psi.T @ F.sum(1) @ B.Psi) * B.dx ** 3,
          (1, 2): (B.Psi.T @ F.sum(0) @ B.Psi) * B.dx ** 3}
    Zt = np.einsum("ak,bl,cm,abc->klm", B.Psi, B.Psi, B.Psi, F) * B.dx ** 3
    pairs = list(Zp.keys())
    S = np.zeros((K - 2, K))
    for i in range(K - 2):
        S[i, i], S[i, i + 1], S[i, i + 2] = 1.0, -2.0, 1.0

    def fg(p):
        th = p[:d * r * K].reshape(d, r, K); lam = p[d * r * K:d * r * K + r]
        Gl = [lam @ th[j] for j in range(d)]
        Gp = {pr: np.einsum("m,mk,ml->kl", lam, th[pr[0]], th[pr[1]]) for pr in pairs}
        Gt = np.einsum("m,ma,mb,mc->abc", lam, th[0], th[1], th[2])
        Rl = [Gl[j] - Zl[j] for j in range(d)]
        Rp = {pr: Gp[pr] - Zp[pr] for pr in pairs}; Rt = Gt - Zt
        val = 0.5 * sum(float(a @ a) for a in Rl) + 0.5 * sum(float((Rp[pr] ** 2).sum()) for pr in pairs) \
            + 0.5 * float((Rt ** 2).sum())
        if gamma > 0:
            val += 0.5 * gamma * sum(float(np.sum((S @ th[j].T) ** 2)) for j in range(d))
        gth = np.zeros((d, r, K)); glm = np.zeros(r)
        for j in range(d):
            gth[j] += lam[:, None] * Rl[j][None, :]; glm += th[j] @ Rl[j]
            if gamma > 0:
                gth[j] += gamma * (S.T @ (S @ th[j].T)).T
        for pr in pairs:
            R = Rp[pr]
            gth[pr[0]] += lam[:, None] * (th[pr[1]] @ R.T)
            gth[pr[1]] += lam[:, None] * (th[pr[0]] @ R)
            glm += np.einsum("mk,kl,ml->m", th[pr[0]], R, th[pr[1]])
        gth[0] += lam[:, None] * np.einsum("abc,mb,mc->ma", Rt, th[1], th[2])
        gth[1] += lam[:, None] * np.einsum("abc,ma,mc->mb", Rt, th[0], th[2])
        gth[2] += lam[:, None] * np.einsum("abc,ma,mb->mc", Rt, th[0], th[1])
        glm += np.einsum("abc,ma,mb,mc->m", Rt, th[0], th[1], th[2])
        return val, np.concatenate([gth.ravel(), glm])

    ts = d * r * K; ub = np.tile(1.0 / B.C_k, d * r)
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
    th = best.x[:ts].reshape(d, r, K); lam = np.clip(best.x[ts:ts + r], 1e-12, None); lam /= lam.sum()
    G = (B.Psi.T @ B.Psi) * B.dx; Ginv = np.linalg.pinv(G)
    u = np.einsum("jmk,nk->jmn", np.einsum("ik,jmk->jmi", Ginv, th), B.Psi)
    u = np.maximum(u, 0); u /= np.maximum(u.sum(2, keepdims=True) * B.dx, 1e-30)
    ctr = np.stack([(u[j] * B.xg[None, :]).sum(1) * B.dx for j in range(d)], 1)  # (r,3)
    return ctr


def run_video3d(cfg_json):
    cfg = json.loads(cfg_json)
    r = int(cfg.get("r", 2)); T = int(cfg.get("T", 10)); sigma = float(cfg.get("sigma", 0.07))
    seed = int(cfg.get("seed", 0)); K = int(cfg.get("K", 8)); N = int(cfg.get("N", 30))
    gamma = float(cfg.get("gamma", 1e-4))
    B = Basis(K=K, N=N)
    F, ctrue = _make_video3d(B, r, T, sigma, seed)
    rec = np.zeros_like(ctrue)
    for t in range(T):
        ctr = _grid_fit3d(F[t], B, r, gamma=gamma, restarts=3, seed=t)
        if t > 0:
            C = ((rec[t - 1][:, None, :] - ctr[None, :, :]) ** 2).sum(-1)
            rs, cs = linear_sum_assignment(C); ctr = ctr[cs]
        rec[t] = ctr
    # temporal repair in 3D
    rec_t = rec.copy()
    for m in range(r):
        for t in range(1, T - 1):
            pred = 0.5 * (rec[t - 1, m] + rec[t + 1, m])
            if np.linalg.norm(rec[t, m] - pred) > 0.6 * sigma:
                rec_t[t, m] = pred
    def err(a):
        e = []
        for t in range(T):
            C = ((ctrue[t][:, None, :] - a[t][None, :, :]) ** 2).sum(-1)
            rs, cs = linear_sum_assignment(C); e.append(float(np.sqrt(C[rs, cs]).mean()))
        return e
    ei, et = err(rec), err(rec_t)
    # 3D point cloud per frame (weighted sample of the voxel density) for volume view
    M = int(cfg.get("M", 650)); rng = np.random.default_rng(7)
    cloud = []
    for t in range(T):
        p = F[t].reshape(-1); p = p / p.sum()
        idx = rng.choice(p.size, size=min(M, p.size), replace=False, p=p)
        ii, jj, kk = np.unravel_index(idx, (N, N, N))
        wt = p[idx] / p[idx].max()
        pts = np.stack([B.xg[ii], B.xg[jj], B.xg[kk], wt], 1).round(4)
        cloud.append(pts.tolist())
    out = {"r": r, "T": T, "true": ctrue.tolist(),
           "rec_indep": rec.tolist(), "rec_temp": rec_t.tolist(),
           "cloud": cloud, "err_indep": ei, "err_temp": et,
           "mean_indep": float(np.mean(ei)), "mean_temp": float(np.mean(et))}
    return json.dumps(out)
