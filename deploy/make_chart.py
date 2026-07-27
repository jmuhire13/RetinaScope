"""Render the Stage 7 load-test scaling chart from the Locust CSV results.
Two panels (no dual-axis): throughput and latency vs replica count."""
import csv
import matplotlib.pyplot as plt

# --- palette (dataviz reference, light mode) ---
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECOND, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

scales = [1, 2, 4]
rps, median_s, p95_s = [], [], []
for n in scales:
    with open(f"deploy/loadtest_results/scale_{n}_stats.csv") as f:
        agg = [r for r in csv.DictReader(f) if r["Name"] == "Aggregated"][0]
    rps.append(float(agg["Requests/s"]))
    median_s.append(float(agg["Median Response Time"]) / 1000.0)
    p95_s.append(float(agg["95%"]) / 1000.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor(SURFACE)
x = range(len(scales))
labels = [f"{n} replica" + ("s" if n > 1 else "") for n in scales]

# Panel 1: throughput (single series, one hue)
bars = ax1.bar(x, rps, width=0.55, color=BLUE, zorder=3)
for xi, v in zip(x, rps):
    ax1.text(xi, v, f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
ax1.set_title("Throughput vs. replica count", color=INK, fontsize=12, fontweight="bold")
ax1.set_ylabel("Requests / second", color=SECOND)
ax1.set_ylim(0, max(rps) * 1.18)

# Panel 2: latency (two series -> grouped bars, legend + direct labels)
w = 0.36
b1 = ax2.bar([xi - w/2 for xi in x], median_s, width=w, color=BLUE, label="Median", zorder=3)
b2 = ax2.bar([xi + w/2 for xi in x], p95_s, width=w, color=ORANGE, label="p95", zorder=3)
for xi, v in zip(x, median_s):
    ax2.text(xi - w/2, v, f"{v:.1f}s", ha="center", va="bottom", color=INK, fontsize=9)
for xi, v in zip(x, p95_s):
    ax2.text(xi + w/2, v, f"{v:.1f}s", ha="center", va="bottom", color=INK, fontsize=9)
ax2.set_title("Response latency vs. replica count", color=INK, fontsize=12, fontweight="bold")
ax2.set_ylabel("Seconds (lower is better)", color=SECOND)
ax2.set_ylim(0, max(p95_s) * 1.18)
ax2.legend(frameon=False, loc="upper right")

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=SECOND)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

fig.suptitle("RetinaScope /predict under a 50-user flood (60s, 1 CPU per replica)",
             color=INK, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("deploy/loadtest_results/scaling_chart.png", dpi=130, facecolor=SURFACE)
print("saved deploy/loadtest_results/scaling_chart.png")
