"""使用 httpx 直接读取一次 vLLM SSE 流式响应。"""

import asyncio
import json
from time import perf_counter

import httpx


URL = "http://localhost:8000/v1/chat/completions"

REQUEST_BODY = {
    "model": "qwen-small",
    "messages": [
        {
            "role": "user",
            "content": "请用一句话解释什么是大模型推理服务。",
        }
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": True,
    "stream_options": {"include_usage": True},
}


async def main():
    """发送一次请求，并完整读取 SSE 和 HTTP 响应。"""

    start = perf_counter()
    first_token_time = None
    done_time = None
    usage = None

    # 创建通用异步 HTTP 客户端。
    async with httpx.AsyncClient(timeout=120.0) as client:

        # 发送 POST 请求，并以流式方式读取响应。
        async with client.stream(
            method="POST",
            url=URL,
            json=REQUEST_BODY,
            headers={"Authorization": "Bearer EMPTY"},
        ) as response:
            response.raise_for_status()

            # 随着数据到达，异步逐行读取 SSE。
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                # SSE 数据行形如：data: {"choices": ...}
                data = line.removeprefix("data:").strip()

                if not data:
                    continue

                if data == "[DONE]":
                    # 记录应用层响应完成时间。
                    done_time = perf_counter()
                    print("\n\nReceived [DONE]")

                    # 不使用 break，继续读取到底层 HTTP 响应结束。
                    continue

                # 将 JSON 字符串转换成 Python 字典。
                event = json.loads(data)

                # 最后的统计事件可能包含 usage。
                if event.get("usage") is not None:
                    usage = event["usage"]

                # 普通文本事件通过 choices 携带增量文本。
                choices = event.get("choices") or []

                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")

                    if content:
                        if first_token_time is None:
                            first_token_time = perf_counter()

                        # 不换行地实时打印每一段新增文本。
                        print(content, end="", flush=True)

            # 到达这里说明 aiter_lines() 已经读取到 HTTP EOF。
            print("HTTP response fully consumed")

    if first_token_time is None:
        raise RuntimeError("Missing first-token timing data")

    if done_time is None:
        raise RuntimeError("Missing SSE [DONE] event")

    if usage is None:
        raise RuntimeError("Missing token usage data")

    ttft_ms = (first_token_time - start) * 1000
    e2e_ms = (done_time - start) * 1000

    print(f"TTFT: {ttft_ms:.3f} ms")
    print(f"E2E: {e2e_ms:.3f} ms")
    print(f"Prompt tokens: {usage['prompt_tokens']}")
    print(f"Completion tokens: {usage['completion_tokens']}")
    print(f"Total tokens: {usage['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
