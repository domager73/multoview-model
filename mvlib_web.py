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
    t0 = time.time(); th_m, lm_m = moments_theta(X, B, r, d, gamma, 5, seed); t_m = time.time() - t0
    t0 = time.time(); th_e, lm_e = em_fit(X, B, r, L, d, 4, seed); t_e = time.time() - t0
    t0 = time.time(); th_h, lm_h = em_from_moments(X, B, r, L, d, gamma, 5, seed); t_h = time.time() - t0
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
