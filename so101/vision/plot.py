"""Plot student success per DAgger / BC iteration (so101/videos/vision_distill.png).

    uv run so101/vision/plot.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# training seeds 0, 1, 2 of each method (seed 0 has no suffix)
RUNS = {"DAgger": ("dagger", "#2a78d6"), "Behaviour cloning": ("bc", "#eb6834")}
SEEDS = ["", "_s1", "_s2"]
TEACHER = 0.925  # SAC expert, 200 evaluation episodes
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def main():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for label, (run, color) in RUNS.items():
        logs = [json.loads((HERE / "runs" / f"{run}{s}" / "log.json").read_text()) for s in SEEDS]
        x = np.mean([[e["samples"] / 1000 for e in log] for log in logs], axis=0)
        y = 100 * np.array([[e["student_success"] for e in log] for log in logs])
        ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=color, alpha=0.18, lw=0)
        ax.plot(x, y.mean(axis=0), color=color, lw=2, marker="o", ms=4,
                label=f"{label} (mean of {len(SEEDS)} seeds, band: min–max)")
    ax.axhline(100 * TEACHER, color=MUTED, lw=1.5, ls="--")
    ax.text(2, 100 * TEACHER + 1.5, "SAC teacher (state, 92.5%)", color=MUTED, fontsize=9)
    ax.set_xlabel("training samples (thousands)", color=INK)
    ax.set_ylabel("student success (%)", color=INK)
    ax.set_title("Camera-only student, 50 evaluation episodes per iteration",
                 color=INK, fontsize=11, loc="left")
    ax.set_ylim(30, 100)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    out = HERE.parent / "videos" / "vision_distill.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
