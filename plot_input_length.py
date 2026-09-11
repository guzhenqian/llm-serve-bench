from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# 读取三轮输入长度实验的汇总结果。
data = pd.read_csv("input_length_summary.csv")
x = data["input_length"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 展示输入长度对 Prefill 延迟的影响。
axes[0, 0].plot(x, data["ttft_mean_ms"], marker="o", label="Mean")
axes[0, 0].plot(x, data["ttft_p95_ms"], marker="o", label="P95")
axes[0, 0].fill_between(
    x,
    data["ttft_p95_ms"] - data["ttft_p95_std"],
    data["ttft_p95_ms"] + data["ttft_p95_std"],
    alpha=0.2,
)
axes[0, 0].set_title("TTFT vs Input Length")
axes[0, 0].set_ylabel("Latency (ms)")
axes[0, 0].legend()

# 展示端到端延迟随输入长度的变化。
axes[0, 1].plot(x, data["e2e_mean_ms"], marker="o", label="Mean")
axes[0, 1].plot(x, data["e2e_p95_ms"], marker="o", label="P95")
axes[0, 1].set_title("E2E Latency vs Input Length")
axes[0, 1].set_ylabel("Latency (ms)")
axes[0, 1].legend()

# 检查 Decode 阶段是否受到输入长度影响。
axes[1, 0].plot(x, data["tpot_mean_ms"], marker="o", label="Mean")
axes[1, 0].plot(x, data["tpot_p95_ms"], marker="o", label="P95")
axes[1, 0].set_title("TPOT vs Input Length")
axes[1, 0].set_ylabel("Time per token (ms)")
axes[1, 0].legend()

# 展示输入长度增加后请求吞吐量的下降趋势。
axes[1, 1].plot(
    x,
    data["request_qps_mean"],
    marker="o",
    color="#C00000",
)
axes[1, 1].set_title("Request Throughput vs Input Length")
axes[1, 1].set_ylabel("Requests/s")

for axis in axes.flat:
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    axis.set_xticklabels(x.astype(str))
    axis.set_xlabel("Configured Input Length (tokens)")
    axis.grid(linestyle="--", alpha=0.4)

fig.suptitle("Input Length Performance Impact", fontsize=14)
fig.tight_layout()

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "input_length_scaling.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Figure saved to {output_path}")
