import asyncio
import csv
import json
from time import perf_counter

import httpx
import numpy as np


# vLLM 提供的 OpenAI 兼容接口。
BASE_URL = "http://localhost:8000/v1"
CHAT_ENDPOINT = f"{BASE_URL}/chat/completions"

MODEL = "qwen-small"

# 依次测试 1、2、4、8、16 个请求同时执行。
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# 每个并发等级总共发送 32 个请求。
REQUESTS_PER_LEVEL = 32

# 正式测试前先运行 3 次，让模型和 CUDA 内核完成预热。
WARMUP_RUNS = 3


async def run_one(client, concurrency, request_id, semaphore):
    """
    发送一个流式请求，并测量该请求的 TTFT、TPOT 和 E2E。

    client:
        所有请求共享的异步 HTTP 客户端。

    concurrency:
        当前实验的目标并发度。

    request_id:
        当前请求的编号。

    semaphore:
        限制同一时刻实际执行的请求数量。
    """

    # 只有获得信号量之后，请求才可以正式开始。
    # 因此排队等待信号量的时间不会被计入单请求延迟。
    async with semaphore:
        start = perf_counter()

        # 在收到第一个有效文本片段时赋值。
        first_token_time = None

        # Token 统计通常出现在最后一个 JSON 数据块中。
        usage = None

        # 用来确认服务端确实发送了 SSE 结束标记。
        received_done = False

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"编号C{concurrency}-R{request_id}："
                        "请列出四条提高大模型推理服务性能的方法，"
                        "每条一句话。"
                    ),
                }
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }

        # client.stream() 发送请求，但不会一次性读取完整响应。
        # response 会随着服务器生成内容而不断收到新的字节。
        async with client.stream(
            method="POST",
            url=CHAT_ENDPOINT,
            json=payload,
        ) as response:
            # 如果服务器返回 4xx 或 5xx，立即抛出异常。
            response.raise_for_status()

            # 按行读取 SSE 响应。
            #
            # 典型内容如下：
            # data: {"choices": [...]}
            #
            # data: [DONE]
            async for line in response.aiter_lines():
                # 空行、注释行和其他非 data 行不需要处理。
                if not line.startswith("data:"):
                    continue

                # 去掉开头的 "data:"，留下 JSON 或 [DONE]。
                data = line[5:].lstrip()

                if data == "[DONE]":
                    received_done = True

                    # 这里不能使用 break。
                    #
                    # [DONE] 只表示大模型内容生成结束，
                    # 不一定表示底层 HTTP 响应已经完全读取完毕。
                    # 使用 continue 后，循环会继续读取到 HTTP EOF。
                    continue

                if not data:
                    continue

                # 将 JSON 字符串解析为 Python 字典。
                event = json.loads(data)

                # 最后一个统计数据块通常包含 usage。
                if event.get("usage") is not None:
                    usage = event["usage"]

                # usage 数据块的 choices 可能是空列表。
                choices = event.get("choices") or []

                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")

                    # 只记录第一个非空文本片段到达的时间。
                    if content and first_token_time is None:
                        first_token_time = perf_counter()

        # 运行到这里，async for 已经读到 HTTP EOF，
        # async with 也已经正常关闭本次响应。
        end = perf_counter()

    # 以下检查可以避免用缺失数据计算出错误指标。
    if not received_done:
        raise RuntimeError(
            f"Request {request_id} did not receive [DONE]"
        )

    if first_token_time is None:
        raise RuntimeError(
            f"Request {request_id} did not receive text content"
        )

    if usage is None:
        raise RuntimeError(
            f"Request {request_id} did not receive usage data"
        )

    completion_tokens = usage["completion_tokens"]

    # E2E：从发送请求到完整读取并关闭 HTTP 响应。
    e2e_ms = (end - start) * 1000

    # TTFT：从发送请求到收到第一个有效文本片段。
    ttft_ms = (first_token_time - start) * 1000

    # TPOT：首 Token 之后，每个后续 Token 的平均生成时间。
    #
    # 第一个 Token 已经包含在 TTFT 中，因此这里减 1。
    # max(..., 1) 用于避免除以 0。
    tpot_ms = (
        (end - first_token_time)
        * 1000
        / max(completion_tokens - 1, 1)
    )

    # 返回当前请求的原始测量结果。
    return {
        "concurrency": concurrency,
        "request_id": request_id,
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "total_tokens": usage["total_tokens"],
    }


async def run_level(client, concurrency):
    """
    完成一个并发等级的测试，并汇总该等级的性能指标。
    """

    # 例如 concurrency=4 时，同时最多只有 4 个请求执行。
    semaphore = asyncio.Semaphore(concurrency)

    # wall_time 测量整个并发等级处理 32 个请求所需的时间。
    start = perf_counter()

    # 创建 32 个协程，并让 asyncio 并发调度它们。
    #
    # 虽然一次创建了 32 个协程，但 semaphore 会限制
    # 同一时刻真正执行的请求数量。
    rows = await asyncio.gather(
        *[
            run_one(
                client,
                concurrency,
                request_id,
                semaphore,
            )
            for request_id in range(REQUESTS_PER_LEVEL)
        ]
    )

    wall_time = perf_counter() - start

    # 从每个请求的结果字典中分别取出三组延迟数据。
    ttft_values = [row["ttft_ms"] for row in rows]
    tpot_values = [row["tpot_ms"] for row in rows]
    e2e_values = [row["e2e_ms"] for row in rows]

    # 所有请求一共生成了多少个输出 Token。
    output_tokens = sum(
        row["completion_tokens"] for row in rows
    )

    summary = {
        "concurrency": concurrency,
        "requests": len(rows),
        "wall_time_s": wall_time,

        # QPS：平均每秒完成的请求数。
        "qps": len(rows) / wall_time,

        # Output TPS：所有请求每秒生成的输出 Token 数。
        "output_tps": output_tokens / wall_time,

        # P50 表示中位数，P95 表示 95% 请求不超过该数值。
        "ttft_p50_ms": np.percentile(ttft_values, 50),
        "ttft_p95_ms": np.percentile(ttft_values, 95),
        "tpot_p50_ms": np.percentile(tpot_values, 50),
        "tpot_p95_ms": np.percentile(tpot_values, 95),
        "e2e_p50_ms": np.percentile(e2e_values, 50),
        "e2e_p95_ms": np.percentile(e2e_values, 95),
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
    """
    创建客户端、预热模型、执行全部并发实验并保存结果。
    """

    # AsyncClient 是异步 HTTP 客户端。
    #
    # 多个 run_one() 协程共享同一个客户端，
    # 因此客户端能够复用连接池中的 TCP 连接。
    async with httpx.AsyncClient(timeout=120.0) as client:
        print(f"Warming up {WARMUP_RUNS} times...")

        # 预热采用串行请求，不写入正式结果文件。
        warmup_semaphore = asyncio.Semaphore(1)

        for index in range(WARMUP_RUNS):
            await run_one(
                client=client,
                concurrency=1,
                request_id=-(index + 1),
                semaphore=warmup_semaphore,
            )

        all_rows = []
        summaries = []

        # 每次完整测试一个并发等级。
        for concurrency in CONCURRENCY_LEVELS:
            rows, summary = await run_level(
                client,
                concurrency,
            )

            # extend 把本等级的 32 个请求结果逐个加入总列表。
            all_rows.extend(rows)

            # append 把本等级的一条汇总结果加入汇总列表。
            summaries.append(summary)

    # 离开 async with 后，HTTP 客户端及连接池已经正常关闭。

    with open(
        "concurrency_requests_httpx.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    with open(
        "concurrency_summary_httpx.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summaries[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(
        "Results saved to concurrency_requests_httpx.csv "
        "and concurrency_summary_httpx.csv"
    )


# asyncio.run() 创建事件循环，执行 main()，最后关闭事件循环。
asyncio.run(main())
