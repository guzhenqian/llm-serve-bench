import asyncio
import csv
from time import perf_counter

import numpy as np
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen-small"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
REQUESTS_PER_LEVEL = 32
WARMUP_RUNS = 3


async def run_one(client, concurrency, request_id, semaphore):
    # semaphore 保证同时执行的请求不超过目标并发度
    async with semaphore:
        start = perf_counter()
        first_token_time = None
        usage = None

        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"编号C{concurrency}-R{request_id}："
                    "请列出四条提高大模型推理服务性能的方法，每条一句话。"
                ),
            }],
            temperature=0.0,
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage

            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content and first_token_time is None:
                    first_token_time = perf_counter()

        end = perf_counter()

    if first_token_time is None or usage is None:
        raise RuntimeError("Missing streaming timing or usage data")

    completion_tokens = usage.completion_tokens
    e2e_ms = (end - start) * 1000
    ttft_ms = (first_token_time - start) * 1000
    tpot_ms = (end - first_token_time) * 1000 / max(completion_tokens - 1, 1)

    return {
        "concurrency": concurrency,
        "request_id": request_id,
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.total_tokens,
    }


async def run_level(client, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    start = perf_counter()

    rows = await asyncio.gather(*[
        run_one(client, concurrency, i, semaphore)
        for i in range(REQUESTS_PER_LEVEL)
    ])

    wall_time = perf_counter() - start
    ttft = [row["ttft_ms"] for row in rows]
    tpot = [row["tpot_ms"] for row in rows]
    e2e = [row["e2e_ms"] for row in rows]
    output_tokens = sum(row["completion_tokens"] for row in rows)

    summary = {
        "concurrency": concurrency,
        "requests": len(rows),
        "wall_time_s": wall_time,
        "qps": len(rows) / wall_time,
        "output_tps": output_tokens / wall_time,
        "ttft_p50_ms": np.percentile(ttft, 50),
        "ttft_p95_ms": np.percentile(ttft, 95),
        "tpot_p50_ms": np.percentile(tpot, 50),
        "tpot_p95_ms": np.percentile(tpot, 95),
        "e2e_p50_ms": np.percentile(e2e, 50),
        "e2e_p95_ms": np.percentile(e2e, 95),
    }

    print(
        f"C={concurrency:2d}: "
        f"QPS={summary['qps']:.2f}, "
        f"Output TPS={summary['output_tps']:.1f}, "
        f"TTFT P95={summary['ttft_p95_ms']:.1f} ms, "
        f"E2E P95={summary['e2e_p95_ms']:.1f} ms"
    )
    return rows, summary


async def main():
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
        timeout=120.0,
    )

    print(f"Warming up {WARMUP_RUNS} times...")
    semaphore = asyncio.Semaphore(1)
    for i in range(WARMUP_RUNS):
        await run_one(client, 1, -(i + 1), semaphore)

    all_rows = []
    summaries = []

    for concurrency in CONCURRENCY_LEVELS:
        rows, summary = await run_level(client, concurrency)
        all_rows.extend(rows)
        summaries.append(summary)

    with open("concurrency_requests.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    with open("concurrency_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    await client.close()
    print("Results saved to concurrency_requests.csv and concurrency_summary.csv")


asyncio.run(main())
