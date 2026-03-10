import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Data ──────────────────────────────────────────────────────────────────────
kernels = ["gemm_01", "gemm_02", "gemm_03", "gemm_04", "gemm_05", "gemm_06", "gemm_07"]
tflops  = [0.04,       0.46,      0.63,      1.37,      2.90,      2.66,      3.05]
pct     = [1.4,        17.9,      24.4,      53.0,      111.8,     102.9,     117.7]

cublas_tflops = 2.59
torch_tflops  = 2.82

labels = [k.replace("gemm_", "K") for k in kernels]  # K01 … K07 (shorter on axis)

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)

DARK_BG   = "#0d1117"
BAR_BASE  = "#238636"
BAR_HI    = "#2ea043"
CUBLAS_C  = "#f78166"
TORCH_C   = "#79c0ff"
TEXT_C    = "#e6edf3"
GRID_C    = "#21262d"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_C,
    "axes.labelcolor":   TEXT_C,
    "xtick.color":       TEXT_C,
    "ytick.color":       TEXT_C,
    "text.color":        TEXT_C,
    "grid.color":        GRID_C,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
})

# ── Chart 1: TFLOPS per kernel ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG)

x = np.arange(len(kernels))
bars = ax.bar(x, tflops, color=BAR_BASE, width=0.6, zorder=3)

# highlight bars that beat cuBLAS
for bar, t in zip(bars, tflops):
    if t >= cublas_tflops:
        bar.set_color(BAR_HI)

ax.axhline(cublas_tflops, color=CUBLAS_C, linewidth=1.4, linestyle="--", zorder=4)
ax.axhline(torch_tflops,  color=TORCH_C,  linewidth=1.4, linestyle="--", zorder=4)

ax.text(len(kernels) - 0.5, cublas_tflops + 0.06, f"cuBLAS  {cublas_tflops} TFLOPS",
        color=CUBLAS_C, fontsize=8.5, ha="right")
ax.text(len(kernels) - 0.5, torch_tflops  + 0.06, f"torch   {torch_tflops} TFLOPS",
        color=TORCH_C,  fontsize=8.5, ha="right")

for bar, t in zip(bars, tflops):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{t:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_C)

ax.set_xticks(x)
ax.set_xticklabels(kernels, fontsize=9)
ax.set_ylabel("TFLOPS  (2048 × 2048 × 2048, float32)", fontsize=9)
ax.set_title("GEMM Kernel Performance", fontsize=12, pad=12)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis="y", zorder=0)
ax.set_ylim(0, max(tflops) * 1.22)

plt.tight_layout()
out1 = os.path.join(OUT, "tflops.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"saved → {out1}")

# ── Chart 2: % of cuBLAS ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG)

bars = ax.bar(x, pct, color=BAR_BASE, width=0.6, zorder=3)
for bar, p in zip(bars, pct):
    if p >= 100:
        bar.set_color(BAR_HI)

ax.axhline(100, color=CUBLAS_C, linewidth=1.4, linestyle="--", zorder=4)
ax.text(len(kernels) - 0.5, 101.5, "cuBLAS  100%",
        color=CUBLAS_C, fontsize=8.5, ha="right")

for bar, p in zip(bars, pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{p:.0f}%", ha="center", va="bottom", fontsize=8, color=TEXT_C)

ax.set_xticks(x)
ax.set_xticklabels(kernels, fontsize=9)
ax.set_ylabel("% of cuBLAS TFLOPS", fontsize=9)
ax.set_title("GEMM Kernel Performance  (relative to cuBLAS)", fontsize=12, pad=12)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis="y", zorder=0)
ax.set_ylim(0, max(pct) * 1.18)

plt.tight_layout()
out2 = os.path.join(OUT, "pct_cublas.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
print(f"saved → {out2}")
