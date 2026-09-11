#!/usr/bin/env bash
set -euo pipefail

# 输入：轮次1、2或3；输出：该轮7种输入长度对应的JSON结果。
run="${1:-}"

if [[ ! "$run" =~ ^[123]$ ]]; then
    echo "Usage: bash run_input_length.sh <1|2|3>"
    exit 1
fi

run_id=$(printf "%02d" "$run")

for input_len in 32 64 128 256 512 1024 2048; do
    seed=$((run * 10000 + input_len))
    echo "=== Run $run_id, input length: $input_len ==="

    vllm bench serve \
      --backend openai-chat \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen-small \
      --endpoint /v1/chat/completions \
      --host 127.0.0.1 \
      --port 8000 \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len 16 \
      --random-range-ratio 0 \
      --num-prompts 20 \
      --num-warmups 3 \
      --request-rate inf \
      --max-concurrency 1 \
      --ignore-eos \
      --temperature 0 \
      --seed "$seed" \
      --percentile-metrics ttft,tpot,e2el \
      --metric-percentiles 50,95 \
      --save-result \
      --save-detailed \
      --result-dir input_length_results \
      --result-filename "input_${input_len}_run_${run_id}.json"
done
