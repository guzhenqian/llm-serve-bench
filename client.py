from time import perf_counter

from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

start = perf_counter()

response = client.chat.completions.create(
    model="qwen-small",
    messages=[
        {
            "role": "user",
            "content": "请用一句话解释什么是大模型推理服务。",
        }
    ],
    temperature=0.2,
    max_tokens=100,
)

latency = perf_counter() - start

print("Answer:", response.choices[0].message.content)
print(f"End-to-end latency: {latency:.3f} s")

if response.usage:
    print("Prompt tokens:", response.usage.prompt_tokens)
    print("Completion tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)
