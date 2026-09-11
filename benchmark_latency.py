import csv
from time import perf_counter

import numpy as np
from openai import OpenAI


WARMUP_RUNS = 3
BENCHMARK_RUNS = 20

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)


def run_once():
    start = perf_counter()
    first_token_time = None
    usage = None

    stream = client.chat.completions.create(
        model="qwen-small",
        messages=[
            {
                "role": "user",
                "content": "请用一句话解释什么是大模型推理服务。",
            }
        ],
        temperature=0.0,
        max_tokens=100,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage

        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content and first_token_time is None:
                first_token_time = perf_counter()

    end = perf_counter()

    if first_token_time is None or usage is None:
        raise RuntimeError("Missing streaming timing or usage data")

    output_tokens = usage.completion_tokens

    if output_tokens <= 1:
        raise RuntimeError("Not enough output tokens to calculate TPOT")

    e2e_ms = (end - start) * 1000
    ttft_ms = (first_token_time - start) * 1000
    tpot_ms = (end - first_token_time) * 1000 / (output_tokens - 1)

    return {
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_tps": 1000 / tpot_ms,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": usage.total_tokens,
    }


print(f"Warming up {WARMUP_RUNS} times...")
for _ in range(WARMUP_RUNS):
    run_once()

results = []

for run_id in range(1, BENCHMARK_RUNS + 1):
    result = run_once()
    result["run"] = run_id
    results.append(result)

    print(
        f"Run {run_id:02d}: "
        f"TTFT={result['ttft_ms']:.2f} ms, "
        f"TPOT={result['tpot_ms']:.3f} ms, "
        f"E2E={result['e2e_ms']:.2f} ms"
    )

fields = [
    "run", "e2e_ms", "ttft_ms", "tpot_ms", "decode_tps",
    "prompt_tokens", "completion_tokens", "total_tokens",
]

with open("latency_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("\n=== Summary ===")

for metric in ["ttft_ms", "tpot_ms", "e2e_ms", "decode_tps"]:
    values = [result[metric] for result in results]
    print(
        f"{metric:12s} "
        f"mean={np.mean(values):.3f}, "
        f"p50={np.percentile(values, 50):.3f}, "
        f"p95={np.percentile(values, 95):.3f}"
    )

print("\nResults saved to latency_results.csv")
