"""Stage 3 figure: out-of-sample error for every day of the year.

Usage: python stage3_plot.py stage3_latent16_by_day.csv
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"


def main(path):
    res = pd.read_csv(path, parse_dates=["date"])
    held = res[res.kind == "held-out day"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, facecolor=SURFACE)
    panels = [("val_rmse", "Held-out quotes"), ("exp_rmse", "Held-out expiries")]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(res.date, res[col], color=BLUE, lw=2, label="Training days")
        ax.scatter(held.date, held[col], s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label="Days the network never trained on")
        ax.set_title(f"{title}: RMSE in vol points", fontsize=11, color=INK, loc="left")
        ax.set_ylim(0, None)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.suptitle("One network for all of 2023: daily out-of-sample error", fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = path.replace("_by_day.csv", "_error_by_day.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
