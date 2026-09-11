import json
import re
from pathlib import Path

import pandas as pd


RESULT_DIR = Path("output_length_results")
FILE_PATTERN = re.compile(r"output_(\d+)_run_(\d+)\.json")


def load_run_results():
    """读取18个正式JSON结果，返回每轮实验对应的数据表。"""
    rows = []

    for path in RESULT_DIR.glob("output_*_run_*.json"):
        match = FILE_PATTERN.fullmatch(path.name)

        # 跳过不符合正式结果命名规则的文件。
        if match is None:
            continue

        output_length = int(match.group(1))
        run_id = int(match.group(2))

        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        completed = result["completed"]

        rows.append({
            "output_length": output_length,
            "run": run_id,
            "successful_requests": completed,
            "failed_requests": result.get("failed", 0),
            "avg_input_tokens": result["total_input_tokens"] / completed,
            "avg_output_tokens": result["total_output_tokens"] / completed,
            "request_qps": result["request_throughput"],
            "output_tps": result["output_throughput"],
            "ttft_mean_ms": result["mean_ttft_ms"],
            "ttft_p95_ms": result["p95_ttft_ms"],
            "tpot_mean_ms": result["mean_tpot_ms"],
            "tpot_p95_ms": result["p95_tpot_ms"],
            "e2e_mean_ms": result["mean_e2el_ms"],
            "e2e_p95_ms": result["p95_e2el_ms"],
        })

    return pd.DataFrame(rows)


def summarize_runs(runs):
    """按照输出长度分组，计算三轮实验的平均值和标准差。"""
    return runs.groupby("output_length", as_index=False).agg(
        runs=("run", "nunique"),
        avg_input_tokens=("avg_input_tokens", "mean"),
        avg_output_tokens=("avg_output_tokens", "mean"),
        request_qps_mean=("request_qps", "mean"),
        output_tps_mean=("output_tps", "mean"),
        output_tps_std=("output_tps", "std"),
        ttft_mean_ms=("ttft_mean_ms", "mean"),
        ttft_p95_ms=("ttft_p95_ms", "mean"),
        tpot_mean_ms=("tpot_mean_ms", "mean"),
        tpot_p95_ms=("tpot_p95_ms", "mean"),
        e2e_mean_ms=("e2e_mean_ms", "mean"),
        e2e_p95_ms=("e2e_p95_ms", "mean"),
    )


def main():
    """检查数据完整性，保存逐轮结果和三轮汇总结果。"""
    runs = load_run_results()
    runs = runs.sort_values(["output_length", "run"])

    counts = runs.groupby("output_length")["run"].nunique()

    if len(runs) != 18 or len(counts) != 6 or not counts.eq(3).all():
        raise RuntimeError(
            "结果不完整：应有6种输出长度，每种长度包含3轮，共18个结果。"
        )

    if runs["failed_requests"].sum() != 0:
        raise RuntimeError("实验结果中存在失败请求，请先检查日志。")

    summary = summarize_runs(runs)

    runs.to_csv("output_length_runs.csv", index=False)
    summary.to_csv("output_length_summary.csv", index=False)

    print("=== Output Length Summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nResults saved to output_length_runs.csv and output_length_summary.csv")


if __name__ == "__main__":
    main()
