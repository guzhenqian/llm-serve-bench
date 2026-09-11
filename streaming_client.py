from time import perf_counter

from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

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

print("Answer: ", end="", flush=True)

for chunk in stream:
    if chunk.usage is not None:
        usage = chunk.usage

    if not chunk.choices:
        continue

    content = chunk.choices[0].delta.content

    if content:
        if first_token_time is None:
            first_token_time = perf_counter()

        print(content, end="", flush=True)

end = perf_counter()
print()

e2e_latency = end - start

print(f"End-to-end latency: {e2e_latency:.3f} s")

if first_token_time is not None:
    ttft = first_token_time - start
    print(f"TTFT: {ttft:.3f} s")

if usage is not None:
    output_tokens = usage.completion_tokens

    print("Prompt tokens:", usage.prompt_tokens)
    print("Completion tokens:", output_tokens)
    print("Total tokens:", usage.total_tokens)

    if first_token_time is not None and output_tokens > 1:
        tpot = (end - first_token_time) / (output_tokens - 1)
        print(f"TPOT: {tpot * 1000:.3f} ms/token")
        print(f"Decode speed: {1 / tpot:.2f} tokens/s")
