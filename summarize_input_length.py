import json
import re
from pathlib import Path
from statistics import mean

import pandas as pd


RESULT_DIR = Path("input_length_results")
FILE_PATTERN = re.compile(r"input_(\d+)_run_(\d+)\.json")


def load_run(path):
    """读取单次JSON结果，返回一行便于汇总的实验数据。"""
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    return {
        "input_length": int(match.group(1)),
        "run": int(match.group(2)),
        "avg_input_tokens": mean(data["input_lens"]),
        "avg_output_tokens": mean(data["output_lens"]),
        "request_qps": data["request_throughput"],
        "output_tps": data["output_throughput"],
        "ttft_mean_ms": data["mean_ttft_ms"],
        "ttft_p95_ms": data["p95_ttft_ms"],
        "tpot_mean_ms": data["mean_tpot_ms"],
        "tpot_p95_ms": data["p95_tpot_ms"],
        "e2e_mean_ms": data["mean_e2el_ms"],
        "e2e_p95_ms": data["p95_e2el_ms"],
    }


# 读取21个正式结果；该模式不会包含冒烟测试文件。
paths = sorted(RESULT_DIR.glob("input_*_run_*.json"))

if len(paths) != 21:
    raise RuntimeError(f"Expected 21 result files, found {len(paths)}")

runs = pd.DataFrame([load_run(path) for path in paths])
runs = runs.sort_values(["input_length", "run"])

run_counts = runs.groupby("input_length")["run"].nunique()
if len(run_counts) != 7 or not run_counts.eq(3).all():
    raise RuntimeError(f"Expected 3 runs for each input length:\n{run_counts}")

# 按输入长度计算三轮平均值，并统计TTFT P95的轮间波动。
summary = (
    runs.groupby("input_length")
    .agg(
        runs=("run", "count"),
        avg_input_tokens=("avg_input_tokens", "mean"),
        avg_output_tokens=("avg_output_tokens", "mean"),
        request_qps_mean=("request_qps", "mean"),
        output_tps_mean=("output_tps", "mean"),
        ttft_mean_ms=("ttft_mean_ms", "mean"),
        ttft_p95_ms=("ttft_p95_ms", "mean"),
        ttft_p95_std=("ttft_p95_ms", "std"),
        tpot_mean_ms=("tpot_mean_ms", "mean"),
        tpot_p95_ms=("tpot_p95_ms", "mean"),
        e2e_mean_ms=("e2e_mean_ms", "mean"),
        e2e_p95_ms=("e2e_p95_ms", "mean"),
    )
    .reset_index()
)

runs.to_csv("input_length_runs.csv", index=False)
summary.to_csv("input_length_summary.csv", index=False)

print("=== Input Length Summary ===")
print(summary.round(2).to_string(index=False))
print("\nResults saved to input_length_runs.csv and input_length_summary.csv")
