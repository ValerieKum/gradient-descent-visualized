from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.viz import (PALETTE, card_figure, caption, save_animation,
                        use_headless_if_saving)

# ---- tunables -------------------------------------------------------------
LEARNING_RATES = [0.02, 0.12, 0.32]   # crawl / glide / overshoot
COLORS_KEY = ["cyan", "hot", "pink"]
START = (-1.9, 1.7)
TOTAL_FRAMES = 120
TRAIL = 18


def loss(x, y):
    """A scenic non-convex surface: a broad bowl with sinusoidal ripples."""
    return (0.5 * (x ** 2 + y ** 2)
            + 1.3 * np.sin(1.7 * x) * np.cos(1.7 * y))


def grad(x, y):
    dx = x + 1.3 * 1.7 * np.cos(1.7 * x) * np.cos(1.7 * y)
    dy = y - 1.3 * 1.7 * np.sin(1.7 * x) * np.sin(1.7 * y)
    return dx, dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="OUT.gif")
    args = ap.parse_args()
    use_headless_if_saving(args.save)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    lim = 2.4
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 200),
                         np.linspace(-lim, lim, 200))
    Z = loss(gx, gy)

    fig, ax = card_figure()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.contourf(gx, gy, Z, levels=24, cmap="magma", alpha=0.9)
    ax.contour(gx, gy, Z, levels=12, colors=PALETTE["bg"], linewidths=0.4,
               alpha=0.4)

    runners = []
    for lr, ckey in zip(LEARNING_RATES, COLORS_KEY):
        c = PALETTE[ckey]
        (trail,) = ax.plot([], [], color=c, lw=1.6, alpha=0.7)
        (dot,) = ax.plot([], [], "o", color=c, ms=8,
                         markeredgecolor=PALETTE["bg"], markeredgewidth=1.2)
        runners.append({"lr": lr, "pos": np.array(START, float),
                        "hist": [START], "trail": trail, "dot": dot})
    txt = caption(ax, "gradient descent · lr "
                  + " ".join(f"{r:.2f}" for r in LEARNING_RATES))

    def render(frame):
        artists = [txt]
        for r in runners:
            x, y = r["pos"]
            dx, dy = grad(x, y)
            r["pos"] = np.array([x - r["lr"] * dx, y - r["lr"] * dy])
            r["pos"] = np.clip(r["pos"], -lim, lim)
            r["hist"].append(tuple(r["pos"]))
            h = np.array(r["hist"][-TRAIL:])
            r["trail"].set_data(h[:, 0], h[:, 1])
            r["dot"].set_data([r["pos"][0]], [r["pos"][1]])
            artists += [r["trail"], r["dot"]]
        return artists

    anim = FuncAnimation(fig, render, frames=TOTAL_FRAMES, interval=50,
                         blit=True)

    if args.save:
        save_animation(anim, args.save, fps=20)
    else:
        plt.show()


if __name__ == "__main__":
    main()
