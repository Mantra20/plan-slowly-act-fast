"""
Summarise results.json written by sim.py: table of the four agents, compute by block,
and a figure of compute at the start of each block.

Usage:  python summarize.py [results.json] [figure.png]
"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
fig_path = sys.argv[2] if len(sys.argv) > 2 else "compute_by_block.png"

d = json.load(open(path))
R, E, S = d["results"], d["meta"]["ep_per_block"], d["meta"]["schedule"]
NAMES = {
    "planif_seul": "Permanent planner",
    "habitude_figee": "Habit frozen after first block",
    "arbitre_sans_mem": "Arbiter without context memory",
    "arbitre_mem_ctx": "Arbiter with context memory",
}

revisit = [i for i, c in enumerate(S) if c in S[:i]]            # context already seen
first = [i for i, c in enumerate(S) if c not in S[:i] and i > 0]  # context seen for the first time
head = lambda blocks: np.concatenate([np.arange(i * E, i * E + 5) for i in blocks])
idx_rev, idx_first, idx_sw = head(revisit), head(first), head(range(1, len(S)))

summ = {}
print("| Agent | Success | Success after a change | Compute per episode |")
print("|---|---|---|---|")
for k, runs in R.items():
    comp = np.array([r["compute"] for r in runs]) / 1000
    suc = np.array([r["success"] for r in runs])
    summ[k] = dict(succ=suc.mean(), succ_sw=suc[:, idx_sw].mean(), comp=comp.mean(),
                   comp_rev=comp[:, idx_rev].mean(), comp_first=comp[:, idx_first].mean(),
                   n_ctx=[r["n_ctx"][-1] for r in runs])
    print(f"| {NAMES[k]} | {suc.mean() * 100:.1f}% | {suc[:, idx_sw].mean() * 100:.1f}% "
          f"| {comp.mean():.0f}k |")

c, m, p = summ["arbitre_sans_mem"], summ["arbitre_mem_ctx"], summ["planif_seul"]
print()
print(f"seeds: {len(next(iter(R.values())))}")
print(f"permanent planner / arbiter with memory, compute: {p['comp'] / m['comp']:.2f}x")
print(f"start of a return (first 5 episodes), without / with memory: "
      f"{c['comp_rev']:.0f}k / {m['comp_rev']:.0f}k = {c['comp_rev'] / m['comp_rev']:.2f}x")
print(f"start of a first visit, without / with memory: {c['comp_first']:.0f}k / {m['comp_first']:.0f}k")
print(f"contexts found by the agent with memory, per seed: {m['n_ctx']}")

print("\nmean compute per episode by block (thousands of model steps)")
print("block             " + "  ".join(f"{i + 1}:C{c_}" for i, c_ in enumerate(S)))
for k, runs in R.items():
    comp = np.array([r["compute"] for r in runs]).mean(0) / 1000
    print(f"{k:18s}" + "  ".join(f"{comp[i * E:(i + 1) * E].mean():4.0f}" for i in range(len(S))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

x = np.arange(len(S))
bars = {k: [np.array([r["compute"] for r in R[k]])[:, i * E:i * E + 5].mean() / 1000 for i in x]
        for k in ("arbitre_sans_mem", "arbitre_mem_ctx")}
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
ax.bar(x - 0.2, bars["arbitre_sans_mem"], 0.38, color="#b9b9c0", label=NAMES["arbitre_sans_mem"])
ax.bar(x + 0.2, bars["arbitre_mem_ctx"], 0.38, color="#7a1f2b", label=NAMES["arbitre_mem_ctx"])
ax.axhline(p["comp"], color="#45454d", lw=1, ls="--")
ax.text(len(S) - 0.5, p["comp"] + 4, f"Permanent planner: {p['comp']:.0f}k", ha="right", fontsize=8,
        color="#45454d")
labels = []
for i, c_ in enumerate(S):
    kind = "start" if i == 0 else ("return" if c_ in S[:i] else "first visit")
    labels.append(f"C{c_}\n{kind}")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("Model steps per episode (thousands)", fontsize=8)
ax.set_title("Compute over the first 5 episodes of each block", fontsize=10, loc="left")
ax.tick_params(axis="y", labelsize=8)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
ax.set_ylim(0, max(p["comp"], max(bars["arbitre_sans_mem"])) * 1.25)
fig.tight_layout()
fig.savefig(fig_path)
print(f"\nfigure written to {fig_path}")
