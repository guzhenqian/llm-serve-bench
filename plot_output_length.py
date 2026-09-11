from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# 读取三轮输出长度实验的汇总结果。
data = pd.read_csv("output_length_summary.csv")
x = data["output_length"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 输出越长，端到端延迟预计近似线性增长。
axes[0, 0].plot(x, data["e2e_mean_ms"], marker="o", label="Mean")
axes[0, 0].plot(x, data["e2e_p95_ms"], marker="o", label="P95")
axes[0, 0].set_title("E2E Latency vs Output Length")
axes[0, 0].set_ylabel("Latency (ms)")
axes[0, 0].legend()

# 检查 Decode 阶段的单 token 延迟是否稳定。
axes[0, 1].plot(x, data["tpot_mean_ms"], marker="o", label="Mean")
axes[0, 1].plot(x, data["tpot_p95_ms"], marker="o", label="P95")
axes[0, 1].set_title("TPOT vs Output Length")
axes[0, 1].set_ylabel("Time per token (ms)")
axes[0, 1].legend()

# 输出吞吐量误差棒表示三轮实验的标准差。
axes[1, 0].errorbar(
    x,
    data["output_tps_mean"],
    yerr=data["output_tps_std"],
    marker="o",
    capsize=4,
    color="#70AD47",
)
axes[1, 0].set_title("Output Token Throughput")
axes[1, 0].set_ylabel("Tokens/s")

# 输出变长后，单个请求耗时增加，因此请求吞吐量下降。
axes[1, 1].plot(
    x,
    data["request_qps_mean"],
    marker="o",
    color="#C00000",
)
axes[1, 1].set_title("Request Throughput")
axes[1, 1].set_ylabel("Requests/s")

for axis in axes.flat:
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    axis.set_xticklabels(x.astype(str))
    axis.set_xlabel("Output Length (tokens)")
    axis.grid(linestyle="--", alpha=0.4)

fig.suptitle("Output Length Performance Impact", fontsize=14)
fig.tight_layout()

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "output_length_scaling.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Figure saved to {output_path}")
