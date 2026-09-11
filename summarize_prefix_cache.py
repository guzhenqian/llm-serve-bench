import csv
import json
from pathlib import Path


# 读取四组实验结果，提取用于对比的核心指标。
experiments = [
    ("on", "cold", "cache_on_cold.json"),
    ("on", "warm", "cache_on_warm.json"),
    ("off", "first", "cache_off_first.json"),
    ("off", "repeat", "cache_off_repeat.json"),
]

rows = []

for cache_status, state, filename in experiments:
    path = Path("prefix_cache_results") / filename

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    rows.append({
        "prefix_cache": cache_status,
        "state": state,
        "request_qps": data["request_throughput"],
        "ttft_mean_ms": data["mean_ttft_ms"],
        "ttft_p95_ms": data["p95_ttft_ms"],
        "tpot_mean_ms": data["mean_tpot_ms"],
        "tpot_p95_ms": data["p95_tpot_ms"],
        "e2e_mean_ms": data["mean_e2el_ms"],
        "e2e_p95_ms": data["p95_e2el_ms"],
    })

output_path = Path("prefix_cache_summary.csv")

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(row)

print(f"\nResults saved to {output_path}")
