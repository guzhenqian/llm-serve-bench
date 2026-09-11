from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# 读取三轮细粒度并发实验结果，并按并发数计算均值和标准差。
files = [
    "concurrency_summary_saturation_fine_01.csv",
    "concurrency_summary_saturation_fine_02.csv",
    "concurrency_summary_saturation_fine_03.csv",
]

runs = pd.concat(
    [pd.read_csv(filename) for filename in files],
    ignore_index=True,
)

summary = runs.groupby("concurrency").agg(
    qps_mean=("qps", "mean"),
    qps_std=("qps", "std"),
    output_tps_mean=("output_tps", "mean"),
    output_tps_std=("output_tps", "std"),
    ttft_p95_mean=("ttft_p95_ms", "mean"),
    ttft_p95_std=("ttft_p95_ms", "std"),
    e2e_p95_mean=("e2e_p95_ms", "mean"),
    e2e_p95_std=("e2e_p95_ms", "std"),
).reset_index()

metrics = [
    ("qps_mean", "qps_std", "Request Throughput", "Requests/s"),
    ("output_tps_mean", "output_tps_std", "Output Token Throughput", "Tokens/s"),
    ("ttft_p95_mean", "ttft_p95_std", "P95 TTFT", "Latency (ms)"),
    ("e2e_p95_mean", "e2e_p95_std", "P95 E2E Latency", "Latency (ms)"),
]

colors = ["#4472C4", "#70AD47", "#ED7D31", "#C00000"]

# 误差棒表示三轮实验的标准差，阴影区域表示观察到的饱和边界。
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for axis, (mean_col, std_col, title, ylabel), color in zip(
    axes.flat, metrics, colors
):
    axis.errorbar(
        summary["concurrency"],
        summary[mean_col],
        yerr=summary[std_col],
        marker="o",
        linewidth=2,
        capsize=4,
        color=color,
    )
    axis.axvspan(64, 80, color="#FFD966", alpha=0.25)
    axis.set_title(title)
    axis.set_xlabel("Concurrency")
    axis.set_ylabel(ylabel)
    axis.set_xticks(summary["concurrency"])
    axis.grid(linestyle="--", alpha=0.4)

fig.suptitle("Concurrency Scaling and Saturation", fontsize=14)
fig.tight_layout()

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "concurrency_scaling.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")

print(summary.to_string(index=False))
print(f"\nFigure saved to {output_path}")
