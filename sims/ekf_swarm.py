"""Pure-IMU dead reckoning vs an EKF using inter-bot range+bearing, on a 10x10 grid.

Three estimators run side by side on the SAME true trajectories and sensor noise:
  imu    - integrate the accelerometer only (dead reckoning)
  ekf    - IMU prediction + range/bearing update from a neighbour's shared estimate;
           the neighbour's own uncertainty is ignored (the "naive" cooperative EKF)
  ekf_ci - same, but the neighbour's position covariance is added to the
           measurement noise, so the receiver is not as overconfident

State per bot: [px, py, vx, vy].  Motion: one random 4-neighbour step per tick.
Run:  python sims/ekf_swarm.py      (writes into figures/ and videos/)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.stats import chi2

# ---- scenario (numbers from the questionnaire) ----------------------------
GRID = 10
N_BOTS = 5          # the questionnaire says 5 in one place and 10 in another
STEPS = 100
RUNS = 100
SIGMA_A = 0.1       # accelerometer std
SIGMA_D = 0.2       # range std
SIGMA_TH = 0.5      # bearing std (rad)
RADIUS = 5.0
P0_STD = 0.1        # how well each bot knows its start
MIN_RANGE = 0.3     # bearing is meaningless when two bots sit on top of each other
ROOT = Path(__file__).resolve().parent.parent

F1 = np.array([[1.0, 1.0], [0.0, 1.0]])           # per axis: p += v ; v += a
B1 = np.array([[1.0], [1.0]])                     # acceleration enters both
F = np.zeros((4, 4)); B = np.zeros((4, 2))
for ax in range(2):                               # ordering [px, py, vx, vy]
    for i in range(2):
        for j in range(2):
            F[2 * i + ax, 2 * j + ax] = F1[i, j]
        B[2 * i + ax, ax] = B1[i, 0]
Q = SIGMA_A**2 * B @ B.T
R = np.diag([SIGMA_D**2, SIGMA_TH**2])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def h(pi, pj):
    d = pj - pi
    return np.array([np.hypot(*d), np.arctan2(d[1], d[0])])


def jac(pi, pj):
    """Jacobian of h wrt the receiver's position (2x2). Wrt the sender it is minus this."""
    dx, dy = pj - pi
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    return np.array([[-dx / r, -dy / r], [dy / r2, -dx / r2]])


def neighbours(p):
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = (p[0] + dx, p[1] + dy)
        if 0 <= q[0] < GRID and 0 <= q[1] < GRID:
            out.append((dx, dy))
    return out


class Filter:
    """Mean x (n,4) and covariance P (n,4,4) for every bot, for one estimator."""

    def __init__(self, x0, rng):
        self.P = np.tile(np.eye(4) * P0_STD**2, (len(x0), 1, 1))
        self.x = x0 + rng.normal(0, P0_STD, x0.shape)

    def predict(self, a_meas):
        self.x = self.x @ F.T + a_meas @ B.T
        self.P = F @ self.P @ F.T + Q

    def update(self, links, inflate):
        """links: list of (receiver, sender, z). All senders use their pre-update estimate."""
        xs = self.x.copy()
        Ps = self.P.copy()
        for i, j, z in links:
            pi, pj = self.x[i, :2], xs[j, :2]
            if np.hypot(*(pj - pi)) < MIN_RANGE:
                continue
            Hp = jac(pi, pj)
            H = np.zeros((2, 4)); H[:, :2] = Hp
            Rk = R + (Hp @ Ps[j, :2, :2] @ Hp.T if inflate else 0)
            y = z - h(pi, pj); y[1] = wrap(y[1])
            S = H @ self.P[i] @ H.T + Rk
            K = self.P[i] @ H.T @ np.linalg.inv(S)
            self.x[i] = self.x[i] + K @ y
            I_KH = np.eye(4) - K @ H
            self.P[i] = I_KH @ self.P[i] @ I_KH.T + K @ Rk @ K.T     # Joseph form


def simulate(seed, record=False):
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, GRID, (N_BOTS, 2))
    p = cells.astype(float); v = np.zeros_like(p)
    x0 = np.hstack([p, v])
    filt = {k: Filter(x0, rng) for k in ("imu", "ekf", "ekf_ci")}
    err = {k: np.zeros((STEPS, N_BOTS)) for k in filt}
    rel = {k: np.zeros((STEPS, N_BOTS)) for k in filt}
    nees = {k: np.zeros((STEPS, N_BOTS)) for k in filt}
    frames = []

    for t in range(STEPS):
        v_new = np.array([rng.choice(len(neighbours(c))) for c in cells.astype(int)])
        v_new = np.array([neighbours(c)[k] for c, k in zip(cells.astype(int), v_new)], float)
        a = v_new - v
        v = v_new
        p = p + v
        cells = p.copy()
        a_meas = a + rng.normal(0, SIGMA_A, a.shape)

        # "one bot can transmit its estimate to the other": for every pair in
        # range, one randomly chosen bot sends and the other receives
        links = []
        for i in range(N_BOTS):
            for j in range(i + 1, N_BOTS):
                if np.hypot(*(p[j] - p[i])) < RADIUS:
                    rx, tx = (i, j) if rng.random() < 0.5 else (j, i)
                    z = h(p[rx], p[tx]) + rng.normal(0, [SIGMA_D, SIGMA_TH])
                    links.append((rx, tx, z))

        for name, f in filt.items():
            f.predict(a_meas)
            if name != "imu":
                f.update(links, inflate=(name == "ekf_ci"))
            e = f.x[:, :2] - p
            err[name][t] = np.hypot(e[:, 0], e[:, 1])
            e_rel = e - e.mean(0)          # error left after removing the swarm's shared drift
            rel[name][t] = np.hypot(e_rel[:, 0], e_rel[:, 1])
            for b in range(N_BOTS):
                nees[name][t, b] = e[b] @ np.linalg.inv(f.P[b, :2, :2]) @ e[b]
        if record:
            frames.append(dict(p=p.copy(), links=[(i, j) for i, j, _ in links],
                               x={k: f.x[:, :2].copy() for k, f in filt.items()},
                               P={k: f.P[:, :2, :2].copy() for k, f in filt.items()}))
    return err, rel, nees, frames


def run_all():
    err = {k: np.zeros((RUNS, STEPS, N_BOTS)) for k in ("imu", "ekf", "ekf_ci")}
    rel = {k: np.zeros((RUNS, STEPS, N_BOTS)) for k in err}
    nees = {k: np.zeros((RUNS, STEPS, N_BOTS)) for k in err}
    for r in range(RUNS):
        e, rl, n, _ = simulate(1000 + r)
        for k in err:
            err[k][r], rel[k][r], nees[k][r] = e[k], rl[k], n[k]
    return err, rel, nees


LABEL = {"imu": "IMU only", "ekf": "EKF (naive)", "ekf_ci": "EKF (neighbour covariance added)"}
COL = {"imu": "tab:red", "ekf": "tab:blue", "ekf_ci": "tab:green"}


def plot(err, rel, nees):
    t = np.arange(1, STEPS + 1)
    # 1. position error: one thin line per run (mean over bots) + bold mean
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, data, title in ((axes[0], err, "Absolute position error"),
                            (axes[1], rel, "Relative error (shared drift removed)")):
        for k in data:
            per_run = data[k].mean(2)
            for row in per_run[:30]:
                ax.plot(t, row, color=COL[k], alpha=0.10, lw=0.8)
            ax.plot(t, per_run.mean(0), color=COL[k], lw=2.6, label=LABEL[k])
        ax.set_yscale("log")
        ax.set_xlabel("timestep")
        ax.set_title(f"{title}, mean over bots ({RUNS} runs)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("error (units, log scale)")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "ekf_error.png", dpi=160); plt.close(fig)

    # 2. consistency: average NEES (2 dof) vs the 95% chi-square band
    n = RUNS * N_BOTS
    lo, hi = chi2.ppf(0.025, 2 * n) / n, chi2.ppf(0.975, 2 * n) / n
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k in nees:
        per_run = nees[k].mean(2)
        for row in per_run[:30]:
            ax.plot(t, row, color=COL[k], alpha=0.10, lw=0.8)
        ax.plot(t, per_run.mean(0), color=COL[k], lw=2.6, label=LABEL[k])
    ax.axhspan(lo, hi, color="0.6", alpha=0.35, label="95% consistent band (around 2)")
    ax.set_yscale("log")
    ax.set_xlabel("timestep"); ax.set_ylabel("average NEES (log scale)")
    ax.set_title("Filter consistency: NEES of the position estimate")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "ekf_consistency.png", dpi=160); plt.close(fig)

    print(f"consistent band for the average NEES (2 dof, N={n}): [{lo:.3f}, {hi:.3f}]")
    for k in err:
        e_end = err[k][:, -20:].mean()
        n_all = nees[k].mean()
        inside = np.mean((nees[k].mean((0, 2)) >= lo) & (nees[k].mean((0, 2)) <= hi))
        r_end = rel[k][:, -20:].mean()
        print(f"{LABEL[k]:36s} abs err (last 20) {e_end:7.3f} | rel err (last 20) {r_end:6.3f} | mean NEES {n_all:8.2f} | steps inside band {100 * inside:5.1f}%")


def video(seed=1003):
    from matplotlib.patches import Ellipse
    _, _, _, fr = simulate(seed, record=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    writer = FFMpegWriter(fps=5)
    cols = plt.cm.tab10(np.arange(N_BOTS))
    with writer.saving(fig, str(ROOT / "videos" / "ekf_demo.mp4"), dpi=100):
        for t, f in enumerate(fr):
            for ax, key in zip(axes, ("imu", "ekf")):
                ax.clear()
                ax.set_xlim(-1.5, GRID + 0.5); ax.set_ylim(-1.5, GRID + 0.5)
                ax.set_aspect("equal"); ax.grid(alpha=0.25)
                ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
                for i, j in f["links"]:
                    ax.plot(*zip(f["p"][i], f["p"][j]), color="0.8", lw=0.8, zorder=0)
                for b in range(N_BOTS):
                    ax.scatter(*f["p"][b], s=110, color=cols[b], edgecolor="k", zorder=3)
                    ax.scatter(*f["x"][key][b], s=70, marker="x", color=cols[b], zorder=4)
                    ax.plot(*zip(f["p"][b], f["x"][key][b]), color=cols[b], lw=0.8, ls=":")
                    w, V = np.linalg.eigh(f["P"][key][b])
                    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
                    ax.add_patch(Ellipse(f["x"][key][b], 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]),
                                         angle=ang, fill=False, ec=cols[b], alpha=0.7))
                ax.set_title(f"{LABEL[key]}   t={t + 1}")
            fig.suptitle("dot = true position, x = estimate, ellipse = 2-sigma belief", fontsize=9)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "videos").mkdir(exist_ok=True)
    err, rel, nees = run_all()
    plot(err, rel, nees)
    video()
