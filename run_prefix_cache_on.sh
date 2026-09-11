#!/usr/bin/env bash
set -euo pipefail

# 输入为固定的2048 tokens；输出两个JSON，分别记录冷请求和缓存命中请求。
mkdir -p prefix_cache_results

for state in cold warm; do
    echo "=== Prefix cache ON: $state ==="

    vllm bench serve \
      --backend openai-chat \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen-small \
      --endpoint /v1/chat/completions \
      --host 127.0.0.1 \
      --port 8000 \
      --dataset-name random \
      --random-input-len 2048 \
      --random-output-len 16 \
      --random-range-ratio 0 \
      --num-prompts 20 \
      --num-warmups 0 \
      --ready-check-timeout-sec 0 \
      --request-rate inf \
      --max-concurrency 1 \
      --ignore-eos \
      --temperature 0 \
      --seed 7001 \
      --percentile-metrics ttft,tpot,e2el \
      --metric-percentiles 50,95 \
      --save-result \
      --save-detailed \
      --result-dir prefix_cache_results \
      --result-filename "cache_on_${state}.json"
done
