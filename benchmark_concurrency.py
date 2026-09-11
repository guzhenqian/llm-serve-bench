"""测试本地 vLLM 推理服务在不同并发度下的性能。

实验流程：
1. 正式测试前预热模型。
2. 分别测试并发度 1、2、4、8、16。
3. 每档并发发送 32 个请求。
4. 计算 TTFT、TPOT、E2E、QPS、Output TPS 和 P50/P95。
5. 保存请求明细和并发档位汇总。

这个版本直接使用 httpx 读取 SSE 流。

关键设计：
收到 data: [DONE] 后只记录模型响应完成时间，不立即退出读取循环。
程序继续读取到底层 HTTP 响应真正结束，使 TCP 连接能够正常返回连接池。
"""

import asyncio
import csv
import json
from time import perf_counter

import httpx
import numpy as np


# =============================================================================
# 实验配置
# =============================================================================

# vLLM 提供的 OpenAI-compatible API 地址。
BASE_URL = "http://localhost:8000/v1"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"

# 启动 vLLM 时通过 --served-model-name 设置的模型服务名称。
MODEL = "qwen-small"

# 依次测试五档并发。
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# 每档并发都执行 32 个正式请求。
REQUESTS_PER_LEVEL = 32

# 正式测试前执行 3 个预热请求。
WARMUP_RUNS = 3

# 单个 HTTP 请求最多等待 120 秒。
REQUEST_TIMEOUT_S = 120.0

# 实验结果文件。
REQUEST_RESULTS_PATH = "concurrency_requests.csv"
SUMMARY_RESULTS_PATH = "concurrency_summary.csv"


def build_request_body(concurrency, request_id):
    """构造一次推理请求需要发送的 JSON 数据。

    参数：
        concurrency: 当前测试的并发档位。
        request_id: 当前请求的编号。

    返回：
        一个 Python 字典。httpx 会将它转换成 JSON 请求体。
    """

    return {
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

        # temperature=0 减少随机采样造成的输出长度波动。
        "temperature": 0.0,

        # 限制最大输出长度，使不同并发档位的工作量尽量接近。
        "max_tokens": 64,

        # 开启流式响应，才能测量首 Token 延迟。
        "stream": True,

        # 要求流式响应末尾提供 Token 使用量统计。
        "stream_options": {"include_usage": True},
    }


async def run_one(client, concurrency, request_id, semaphore):
    """执行一次流式推理请求，并返回该请求的性能指标。

    参数：
        client:
            共享的 httpx.AsyncClient。
            它负责异步 HTTP 请求和底层连接池管理。

        concurrency:
            当前实验设置的目标并发度。

        request_id:
            当前并发档位中的请求编号。

        semaphore:
            asyncio 信号量，用于限制同时执行的请求数量。

    返回：
        一个字典，包含当前请求的延迟和 Token 统计。
    """

    # 每档测试会一次性创建 32 个请求协程。
    # 只有成功取得 semaphore 许可的请求才能进入这个代码块。
    async with semaphore:

        # start 位于信号量内部。
        # 因此单请求 E2E 不包括在客户端等待信号量的排队时间。
        start = perf_counter()

        # 收到第一个有效文本 Token 时赋值。
        first_token_time = None

        # 收到 SSE 的 data: [DONE] 时赋值。
        done_time = None

        # Token 统计通常出现在流式响应末尾。
        usage = None

        request_body = build_request_body(
            concurrency=concurrency,
            request_id=request_id,
        )

        # client.stream() 不会等待完整响应后再一次性返回。
        # 它允许程序随着服务器发送数据而逐步读取响应。
        async with client.stream(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            json=request_body,
        ) as response:

            # 如果服务器返回 4xx 或 5xx，立即抛出异常。
            # 这样可以避免将失败请求写入性能结果。
            response.raise_for_status()

            # aiter_lines() 异步逐行读取 SSE 响应。
            #
            # 等待网络数据时，当前协程会暂时交出控制权，
            # asyncio 事件循环就可以继续执行其他并发请求。
            async for line in response.aiter_lines():

                # SSE 的有效数据行以 "data:" 开头。
                # 空行或者其他类型的 SSE 字段暂时跳过。
                if not line.startswith("data:"):
                    continue

                # 删除开头的 "data:"，保留真正的数据内容。
                data = line.removeprefix("data:").strip()

                if not data:
                    continue

                if data == "[DONE]":
                    # [DONE] 只表示 OpenAI-compatible 应用层内容结束。
                    # 它不一定表示底层 HTTP 响应已经完成收尾。
                    done_time = perf_counter()

                    # 这里必须使用 continue，不能使用 break。
                    #
                    # continue:
                    #   继续读取 HTTP 响应，直到真正到达 EOF。
                    #
                    # break:
                    #   立即结束循环，可能留下未读取的 HTTP 结束字节。
                    continue

                # 普通 data 行保存的是 JSON 字符串。
                # json.loads() 将字符串解析成 Python 字典。
                event = json.loads(data)

                # 开启 include_usage 后，末尾事件会包含 usage。
                # 大部分文本事件没有 usage，所以要先判断。
                if event.get("usage") is not None:
                    usage = event["usage"]

                # 文本事件通常包含 choices。
                #
                # 只包含 usage 的末尾事件可能是：
                # {"choices": [], "usage": {...}}
                #
                # 因此读取 choices[0] 前必须检查 choices 是否为空。
                choices = event.get("choices") or []

                if choices:
                    # delta 表示这次流式事件新增加的内容。
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")

                    # 只有第一次收到非空文本时才记录时间。
                    # 后续 Token 不会覆盖 first_token_time。
                    if content and first_token_time is None:
                        first_token_time = perf_counter()

            # 只有 aiter_lines() 读取到 HTTP 响应结束后，
            # 程序才会到达这里。
            #
            # 离开 async with 后，这条 HTTP 连接可以安全返回连接池。

    # 分别检查三类关键数据，便于出现问题时准确定位。
    if first_token_time is None:
        raise RuntimeError("Missing first-token timing data")

    if usage is None:
        raise RuntimeError("Missing token usage data")

    if done_time is None:
        raise RuntimeError("Missing SSE [DONE] event")

    completion_tokens = usage["completion_tokens"]

    # E2E：从请求开始到收到 [DONE] 的总耗时。
    e2e_ms = (done_time - start) * 1000

    # TTFT：从请求开始到收到第一个有效文本 Token 的耗时。
    ttft_ms = (first_token_time - start) * 1000

    # 从第一个 Token 到最后一个 Token 之间，
    # 对 N 个 Token 来说共有 N-1 个时间间隔。
    #
    # max(..., 1) 防止输出不足两个 Token 时除以零。
    tpot_ms = (
        (done_time - first_token_time)
        * 1000
        / max(completion_tokens - 1, 1)
    )

    # 一个字典代表一条请求级实验记录。
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
    """执行一个并发档位，并计算这一档的汇总指标。

    返回：
        rows:
            当前档位下 32 个请求的明细字典。

        summary:
            当前档位的 QPS、Output TPS 和延迟分位数。
    """

    # 例如 concurrency=4 时，最多允许 4 个请求同时执行。
    semaphore = asyncio.Semaphore(concurrency)

    # batch_start 在全部请求创建前记录。
    # 因此 wall_time 包含整批请求的执行时间和客户端排队时间。
    batch_start = perf_counter()

    # 创建 32 个协程对象。
    #
    # 调用 async 函数不会立刻完整执行它，
    # 而是先得到一个描述异步工作的协程对象。
    request_coroutines = [
        run_one(
            client=client,
            concurrency=concurrency,
            request_id=request_id,
            semaphore=semaphore,
        )
        for request_id in range(REQUESTS_PER_LEVEL)
    ]

    # * 将列表展开：
    #
    # asyncio.gather(*request_coroutines)
    #
    # 相当于：
    #
    # asyncio.gather(coroutine_1, coroutine_2, ...)
    #
    # gather 等待全部请求完成，并返回所有 run_one() 的结果。
    rows = await asyncio.gather(*request_coroutines)

    wall_time = perf_counter() - batch_start

    # rows 是“由字典组成的列表”。
    # 每个字典保存一个请求的测量结果。
    #
    # 下面三个列表推导式分别抽取一种延迟指标，
    # 供 NumPy 计算 P50 和 P95。
    ttft = [row["ttft_ms"] for row in rows]
    tpot = [row["tpot_ms"] for row in rows]
    e2e = [row["e2e_ms"] for row in rows]

    # 累加这一档所有请求生成的输出 Token 数。
    output_tokens = sum(
        row["completion_tokens"]
        for row in rows
    )

    summary = {
        "concurrency": concurrency,

        # 当前档位成功完成的请求数。
        "requests": len(rows),

        # 完成这一档全部请求所需的实际时间。
        "wall_time_s": wall_time,

        # QPS = 完成请求数 / 整档耗时。
        "qps": len(rows) / wall_time,

        # Output TPS = 全部输出 Token 数 / 整档耗时。
        "output_tps": output_tokens / wall_time,

        # P50 表示中位数，反映典型请求。
        # P95 表示 95% 请求低于该数值，反映尾延迟。
        "ttft_p50_ms": np.percentile(ttft, 50),
        "ttft_p95_ms": np.percentile(ttft, 95),
        "tpot_p50_ms": np.percentile(tpot, 50),
        "tpot_p95_ms": np.percentile(tpot, 95),
        "e2e_p50_ms": np.percentile(e2e, 50),
        "e2e_p95_ms": np.percentile(e2e, 95),
    }

    # 每完成一档，打印最重要的吞吐和尾延迟指标。
    print(
        f"C={concurrency:2d}: "
        f"QPS={summary['qps']:.2f}, "
        f"Output TPS={summary['output_tps']:.1f}, "
        f"TTFT P95={summary['ttft_p95_ms']:.1f} ms, "
        f"E2E P95={summary['e2e_p95_ms']:.1f} ms"
    )

    return rows, summary


def save_rows_to_csv(path, rows):
    """将“由字典组成的列表”保存为 CSV 文件。"""

    if not rows:
        raise RuntimeError(f"Cannot write empty results to {path}")

    # 第一个字典的键作为 CSV 表头。
    fieldnames = list(rows[0].keys())

    # newline="" 可以避免 CSV 出现多余空行。
    # encoding="utf-8" 使文件能够安全保存中文。
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


async def main():
    """组织客户端创建、预热、正式测试和结果保存。"""

    all_rows = []
    summaries = []

    # HTTP/1.1 中，一条连接同一时刻通常处理一个活动响应。
    # 连接池上限设置为最高并发度，保证并发 16 时有足够连接。
    max_connections = max(CONCURRENCY_LEVELS)

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )

    # AsyncClient 是异步 HTTP 客户端。
    #
    # 整个实验共享一个客户端：
    # 1. 多个请求可以共享连接池。
    # 2. 空闲 TCP 连接可以被后续请求复用。
    # 3. 不需要为每个请求重新创建客户端。
    #
    # async with 可以保证实验结束或发生异常时正确关闭客户端。
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer EMPTY"},
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S),
        limits=limits,
    ) as client:

        print(f"Warming up {WARMUP_RUNS} times...")

        # 预热使用并发度 1，三个请求依次执行。
        # 预热结果不加入正式实验数据。
        warmup_semaphore = asyncio.Semaphore(1)

        for warmup_index in range(WARMUP_RUNS):
            # 负数编号用于在日志中区分预热请求。
            warmup_request_id = -(warmup_index + 1)

            await run_one(
                client=client,
                concurrency=1,
                request_id=warmup_request_id,
                semaphore=warmup_semaphore,
            )

        # 不同并发档位之间依次执行。
        # 每个档位内部的 32 个请求并发执行。
        for concurrency in CONCURRENCY_LEVELS:
            rows, summary = await run_level(
                client=client,
                concurrency=concurrency,
            )

            # extend 将本档的 32 个请求字典逐个加入总列表。
            all_rows.extend(rows)

            # append 将当前档位的一个汇总字典加入列表。
            summaries.append(summary)

    # 离开 async with 后，客户端和连接池已经正确关闭。
    # 所有测试成功完成后，再将数据写入 CSV。
    save_rows_to_csv(
        path=REQUEST_RESULTS_PATH,
        rows=all_rows,
    )

    save_rows_to_csv(
        path=SUMMARY_RESULTS_PATH,
        rows=summaries,
    )

    print(
        f"Results saved to {REQUEST_RESULTS_PATH} "
        f"and {SUMMARY_RESULTS_PATH}"
    )


# 只有直接执行下面的命令时才运行 main()：
#
# python benchmark_concurrency.py
#
# 如果未来从其他文件 import 本模块，main() 不会意外自动运行。
if __name__ == "__main__":
    asyncio.run(main())
