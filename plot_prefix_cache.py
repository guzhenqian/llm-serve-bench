from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# 读取汇总结果，并为四种实验状态生成显示名称。
data = pd.read_csv("prefix_cache_summary.csv")
data["label"] = [
    "Cache ON\nCold",
    "Cache ON\nWarm",
    "Cache OFF\nFirst",
    "Cache OFF\nRepeat",
]

metrics = [
    ("ttft_mean_ms", "Mean TTFT", "Latency (ms)"),
    ("e2e_mean_ms", "Mean E2E Latency", "Latency (ms)"),
    ("request_qps", "Request Throughput", "Requests/s"),
]

colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"]

# 分别展示首 token 延迟、端到端延迟和请求吞吐量。
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for axis, (column, title, ylabel) in zip(axes, metrics):
    bars = axis.bar(data["label"], data[column], color=colors)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.bar_label(bars, fmt="%.2f", padding=3)

fig.suptitle("Prefix Cache Performance Comparison", fontsize=14)
fig.tight_layout()

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "prefix_cache_comparison.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Figure saved to {output_path}")
