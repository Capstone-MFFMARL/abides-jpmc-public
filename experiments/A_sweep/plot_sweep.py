import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "sweep_results.csv"))
cols = [("runtime_s", "Runtime (s)"), ("messages", "Messages"),
        ("spread", "Bid-ask spread (cents)"), ("depth", "Depth (top-5, both sides)"),
        ("vol_bps", "Mid-price vol (bps / 10 s)")]
g = df.groupby("num_mm")[[c for c, _ in cols]].agg(["mean", "std"])
print(g.round(2).to_string())
g.round(3).to_csv(os.path.join(here, "sweep_summary.csv"))

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (c, title) in zip(axes, cols):
    ax.errorbar(g.index, g[(c, "mean")], yerr=g[(c, "std")], fmt="o-", capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(list(g.index))
    ax.set_xticklabels([str(i) for i in g.index])
    ax.set_title(title)
    ax.set_xlabel("num_mm")
    ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(here, "abides_sweep.png"), dpi=200)
print("saved abides_sweep.png")
