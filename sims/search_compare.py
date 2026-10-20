"""BFS vs DFS vs Dijkstra on a 5-layer binary tree, as a physical agent.

The agent starts at the root and can only move along one edge per timestep.
Each algorithm decides the *order* in which nodes get visited; the agent then
walks (including backtracking) from one node to the next in that order, and we
count every edge it crosses until it stands on the target.

"""
import heapq
import random
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

LAYERS = 5                       # 1 + 2 + 4 + 8 + 16 nodes
N_NODES = 2**LAYERS - 1          # 31
TRIALS = 200
SEED = 7
ROOT = Path(__file__).resolve().parent.parent


# node i has children at positions 2i+1 and 2i+2 
def children(i):
    return [c for c in (2 * i + 1, 2 * i + 2) if c < N_NODES]


def parent(i):
    return (i - 1) // 2


def depth(i):
    return (i + 1).bit_length() - 1


def tree_path(a, b):
    """Nodes to walk from a to b (excluding a), going up to the common ancestor."""
    up, down = [], []
    while a != b:
        if depth(a) >= depth(b):
            a = parent(a)
            up.append(a)
        else:
            down.append(b)
            b = parent(b)
    return up + down[::-1]


def tree_dist(a, b):
    return len(tree_path(a, b))


# --- the three search orders ---
def bfs_order():
    order, q = [], deque([0])
    while q:
        n = q.popleft()
        order.append(n)
        q.extend(children(n))
    return order


def dfs_order():
    order, stack = [], [0]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(reversed(children(n)))   # so the left child is popped first
    return order


def dijkstra_order():
    # every edge costs 1, so this is BFS with a heap; the counter breaks ties
    order, heap, count = [], [(0, 0, 0)], 1
    seen = {0}
    while heap:
        d, _, n = heapq.heappop(heap)
        order.append(n)
        for c in children(n):
            if c not in seen:
                seen.add(c)
                heapq.heappush(heap, (d + 1, count, c))
                count += 1
    return order


ALGOS = {"BFS": bfs_order(),
         "DFS": dfs_order(),
        "Dijkstra": dijkstra_order()}


def walk(order, target):
    """Positions of the agent at t = 0, 1, 2, ... until it reaches the target."""
    pos, here = [0], 0
    for node in order:
        for step in tree_path(here, node):
            pos.append(step)
            here = step
            if here == target:
                return pos
    raise RuntimeError("target never reached")


# --- numbers ---
def summary():
    print(f"{'algorithm':10s} {'mean moves':>11s} {'worst':>6s}   (all {N_NODES - 1} possible targets)")
    for name, order in ALGOS.items():
        t = [len(walk(order, tg)) - 1 for tg in range(1, N_NODES)]
        print(f"{name:10s} {np.mean(t):11.2f} {max(t):6d}")


# --- spaghetti plot ---
def spaghetti():
    rng = random.Random(SEED)
    targets = [rng.randrange(1, N_NODES) for _ in range(TRIALS)]
    colours = {"BFS": "tab:blue", "DFS": "tab:orange", "Dijkstra": "tab:green"}
    curves = {k: [] for k in ALGOS}
    for tg in targets:
        for name, order in ALGOS.items():
            p = walk(order, tg)
            curves[name].append([tree_dist(n, tg) for n in p])

    horizon = max(len(c) for cs in curves.values() for c in cs)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    for name, cs in curves.items():
        padded = np.zeros((len(cs), horizon))
        for r, c in enumerate(cs):
            padded[r, : len(c)] = c
        if name == "Dijkstra":
            ax.plot(padded.mean(0), color="k", lw=1.4, ls="--",
                    label="Dijkstra (same curve as BFS)")
            continue
        for row in padded[:40]:
            ax.plot(row, color=colours[name], alpha=0.12, lw=0.8)
        ax.plot(padded.mean(0), color=colours[name], lw=3.2, label=f"{name} (mean of {len(cs)})")
    ax.set_xlabel("timestep (one edge crossed per step)")
    ax.set_ylabel("edges between agent and target")
    ax.set_title("Distance to target over time")
    ax.legend()
    ax.grid(alpha=0.3)

    done = [[len(c) - 1 for c in cs] for cs in curves.values()]
    bx.boxplot(done, tick_labels=list(curves), showmeans=True)
    bx.set_ylabel("timesteps to reach target")
    bx.set_title("Completion time")
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "spaghetti_search.png", dpi=160)
    plt.close(fig)


# --- video ---
def layout():
    xy = {}
    for n in range(N_NODES):
        d = depth(n)
        k = n - (2**d - 1)
        xy[n] = ((k + 0.5) / 2**d, -d)
    return xy


def video(target=22):
    xy = layout()
    paths = {name: walk(order, target) for name, order in ALGOS.items()}
    frames = max(len(p) for p in paths.values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    writer = FFMpegWriter(fps=4)
    with writer.saving(fig, str(ROOT / "videos" / "search_demo.mp4"), dpi=110):
        for t in range(frames):
            for ax, (name, p) in zip(axes, paths.items()):
                ax.clear()
                ax.axis("off")
                for n in range(N_NODES):
                    for c in children(n):
                        ax.plot(*zip(xy[n], xy[c]), color="0.75", lw=1, zorder=1)
                seen = set(p[: min(t, len(p) - 1) + 1])
                for n in range(N_NODES):
                    ax.scatter(*xy[n], s=150, color="tab:blue" if n in seen else "white",
                               edgecolor="k", zorder=2)
                ax.scatter(*xy[target], s=380, marker="*", color="gold", edgecolor="k", zorder=3)
                here = p[min(t, len(p) - 1)]
                ax.scatter(*xy[here], s=230, color="tab:red", edgecolor="k", zorder=4)
                done = t >= len(p) - 1
                ax.set_title(f"{name}: {len(p) - 1} moves" + (" (done)" if done else f"  t={t}"))
            fig.tight_layout()
            writer.grab_frame()
        for _ in range(8):
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "videos").mkdir(exist_ok=True)
    summary()
    spaghetti()
    video()
