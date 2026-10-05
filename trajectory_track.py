"""Joint trajectory tracking: carry bodies through overlaps using a motion model.

Per-frame moment estimates merge when bodies coincide.  Here we instead build
r smooth trajectories: alternate (a) assigning each frame's components to
trajectories and (b) fitting a low-order polynomial to each trajectory, using
all frames (including the merged ones only as soft assignments).  Positions at
merged frames are then *predicted* by the motion model, not read off the frame.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import mvlib_web as m
import repulsion_test as rt


def make_cross(B, r, T=14, sigma=0.06, seed=0):
    rg = np.random.default_rng(seed)
    p0 = rg.uniform(0.2, 0.8, (r, 2))
    p1 = 1.0 - p0                                  # through the centre, random angles
    F = np.zeros((T, B.N, B.N)); gt = np.zeros((T, r, 2))
    g = lambda v, c: np.exp(-(v - c) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    for t in range(T):
        a = t / (T - 1)
        for k in range(r):
            c = (1 - a) * p0[k] + a * p1[k]; gt[t, k] = c
            F[t] += (1.0 / r) * np.outer(g(B.xg, c[0]), g(B.xg, c[1]))
        F[t] /= F[t].sum() * B.dx * B.dx
    return F, gt


def per_frame_centers(B, F, r, restarts=4):
    C = []
    for t in range(len(F)):
        c = rt.fit(F[t], B, r, gamma_div=0.0, restarts=restarts, seed=t)
        C.append(c)
    return C


def _fit_traj(ts, pts, deg=1):
    """Least-squares polynomial fit of pts (n,2) at times ts (n,)."""
    A = np.vander(ts, deg + 1)
    coef, *_ = np.linalg.lstsq(A, pts, rcond=None)
    return coef


def _pred(coef, t):
    return np.array([np.vander([t], coef.shape[0])[0] @ coef[:, k] for k in range(2)])


def track_trajectories(C, r, T, deg=1, iters=6):
    # init with constant velocity from the first 3 (well-separated) frames
    traj_t = list(range(T))
    assign = [None] * T
    # initialise each body's trajectory from frames 0,1 (nearest match forward)
    order = [np.arange(r)]
    for t in range(1, min(3, T)):
        Cs = ((C[t - 1][order[-1]][:, None, :] - C[t][None, :, :]) ** 2).sum(-1)
        _, col = linear_sum_assignment(Cs)
        order.append(col)
    init_pts = {}
    for k in range(r):
        pts = [C[t][order[t][k]] for t in range(len(order))]
        coef = _fit_traj(np.arange(len(order)), np.array(pts), deg)
        init_pts[k] = coef
    coefs = [init_pts[k] for k in range(r)]
    for _ in range(iters):
        assign = [None] * T
        for t in range(T):
            pred = np.array([_pred(coefs[k], t) for k in range(r)])
            cost = ((pred[:, None, :] - C[t][None, :, :]) ** 2).sum(-1)
            rs, cs = linear_sum_assignment(cost)
            assign[t] = cs  # component index for each trajectory
        # refit each trajectory on its assigned points (all frames)
        for k in range(r):
            ts, pts = [], []
            for t in range(T):
                ts.append(t); pts.append(C[t][assign[t][k]])
            coefs[k] = _fit_traj(np.array(ts), np.array(pts), deg)
    return np.array([[ _pred(coefs[k], t) for k in range(r)] for t in range(T)])


def rmse(rec, gt):
    e = []
    for t in range(len(gt)):
        C = ((gt[t][:, None, :] - rec[t][None, :, :]) ** 2).sum(-1)
        rs, cs = linear_sum_assignment(C); e.append(np.sqrt(C[rs, cs]))
    return float(np.concatenate(e).mean())


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print(f"{'r':>2} | {'per-frame':>10} | {'joint traj':>10}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, r in enumerate([2, 3, 4]):
        B = m.Basis(K=max(13, 4 * r + 1), N=64)
        F, gt = make_cross(B, r, T=14, sigma=0.08)
        C = per_frame_centers(B, F, r)
        # per-frame (with simple nearest tracking)
        rec0 = np.zeros_like(gt)
        for t in range(len(F)):
            c = C[t]
            if t > 0:
                _, col = linear_sum_assignment(((rec0[t - 1][:, None, :] - c[None, :, :]) ** 2).sum(-1)); c = c[col]
            rec0[t] = c
        rec1 = track_trajectories(C, r, len(F), deg=1)
        print(f"{r:>2} | {rmse(rec0, gt):10.4f} | {rmse(rec1, gt):10.4f}   (K={B.K})")
        for k in range(r):
            ax[i].plot(gt[:, k, 0], gt[:, k, 1], "k-o", ms=2)
            ax[i].plot(rec0[:, k, 0], rec0[:, k, 1], "--s", ms=2, alpha=.6)
            ax[i].plot(rec1[:, k, 0], rec1[:, k, 1], "-^", ms=2)
        ax[i].set_title(f"r={r}: black=GT, blue=per-frame, orange=joint traj")
    plt.tight_layout(); plt.savefig("trajectory_track_result.png", dpi=130)
    print("saved trajectory_track_result.png")
