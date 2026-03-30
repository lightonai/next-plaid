"""Generate benchmark bar plot for README."""

import matplotlib.pyplot as plt
import numpy as np

# Data: train time in seconds for 100K vectors, 128d, 25 iterations
# H100 for CUDA/GPU backends, Apple Silicon for Metal GPU
# Ordered slowest to fastest (by k=256 time)
backends = [
    "fastkmeans-rs\nCPU (no BLAS)",
    "fastkmeans-rs\nCPU (OpenBLAS)",
    "fastkmeans-rs\nCPU (MKL)",
    "fast-kmeans\nCPU",
    "fastkmeans-rs\nMetal GPU",
    "fast-kmeans\nGPU",
    "fastkmeans-rs\nGPU",
    "flash-kmeans\nGPU",
]

data = {
    256:  [2.678, 2.089, 0.790, 0.677, 0.237, 0.152, 0.127, 0.023],
    512:  [4.788, 3.723, 1.162, 1.108, 0.423, 0.154, 0.135, 0.011],
    1024: [8.784, 7.078, 2.096, 2.010, 0.735, 0.741, 0.159, 0.012],
}

colors = {
    256:  "#4C78A8",
    512:  "#F58518",
    1024: "#E45756",
}

fig, ax = plt.subplots(figsize=(15, 5))

n_backends = len(backends)
n_ks = len(data)
bar_width = 0.25
x = np.arange(n_backends)

for i, (k, times) in enumerate(data.items()):
    offset = (i - n_ks / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, times, bar_width, label=f"k={k}", color=colors[k], edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, times):
        if t < 0.1:
            label = f"{t*1000:.0f}ms"
        else:
            label = f"{t:.2f}s"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

ax.set_ylabel("Train time (s)", fontsize=12)
ax.set_title("K-Means Training — 100K vectors × 128d, 25 iterations (log scale)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(backends, fontsize=9)
# Bold only fastkmeans-rs labels
for tick_label in ax.get_xticklabels():
    if tick_label.get_text().startswith("fastkmeans-rs"):
        tick_label.set_fontweight("bold")
ax.legend(fontsize=10, loc="upper left")
ax.set_yscale("log")
ax.set_ylim(0.005, 15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}s" if v >= 1 else f"{v*1000:.0f}ms" if v < 0.1 else f"{v:.1f}s"))
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("assets/benchmark.png", dpi=180, bbox_inches="tight")
print("Saved assets/benchmark.png")
