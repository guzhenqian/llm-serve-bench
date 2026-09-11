# 大模型推理服务性能实验记录

## 统一实验环境

- GPU：NVIDIA GeForce RTX 4070 Ti 12GB
- 操作系统：WSL2 Ubuntu
- 推理框架：vLLM 0.27.1+cu129
- 模型：Qwen/Qwen2.5-0.5B-Instruct
- PyTorch：2.13.0+cu129
- CUDA：12.9
- 数据类型：BF16
- 最大模型长度（max model length）：4096
- GPU 显存利用率参数（gpu memory utilization）：0.70
- 服务地址：`http://localhost:8000`
- 测试网络：本机回环网络（local loopback network）


## 1. 单请求延迟基线实验

### 实验目的

在并发数为 1 的情况下，测量单个请求的基础延迟和解码速度，为后续并发实验提供对照基线。

### 实验方法

- 测试脚本：`benchmark_latency.py`
- 预热请求数：3
- 正式请求数：20
- 并发数（concurrency）：1
- 输出方式：流式输出（streaming）
- temperature：0
- 每个请求的 Prompt tokens：39
- 每个请求的 Completion tokens：40

预热请求不计入正式统计，用于减少模型首次运行、CUDA 初始化和缓存建立对结果的影响。

### 实验结果

- TTFT mean：9.426 ms
- TTFT P95：12.038 ms
- TPOT mean：2.875 ms/token
- TPOT P95：2.884 ms/token
- E2E mean：121.536 ms
- E2E P95：123.943 ms
- 单请求解码速度：347.876 token/s

其中：

- TTFT（Time To First Token）：从请求发出到收到第一个有效内容 token 的时间。
- TPOT（Time Per Output Token）：生成后续每个输出 token 的平均耗时。
- E2E（End-to-End Latency）：从请求发出到完整响应接收结束的总时间。
- P95：95% 的请求延迟不超过该数值，用于观察尾部延迟。

### 实验结论

在单请求、模型已经完成预热的情况下：

- 首 token 延迟约为 9.4 ms。
- 后续 token 的平均生成耗时约为 2.88 ms/token。
- 单请求解码速度约为 348 token/s。
- 20 次测试结果差异较小，说明单请求性能较为稳定。

这些结果可以作为后续并发测试的低负载性能基线。

### 局限性

- 使用的模型只有 0.5B，不能代表大型模型的推理性能。
- 测试在本机回环网络中进行，不包含远程网络延迟。
- 模型已经预热，因此结果不代表冷启动延迟。
- 请求内容和输出长度基本固定，工作负载比较单一。
- 相同 Prompt 可能受前缀缓存等机制影响。


## 2. 并发扩展基线实验

### 实验目的

观察并发请求数逐渐增加时，vLLM 服务的吞吐量和延迟如何变化，并判断并发数 16 以内是否出现性能饱和。

### 实验方法

- 测试脚本：`benchmark_concurrency_httpx.py`
- HTTP 客户端：HTTPX
- SSE 响应读取方式：持续读取到 HTTP 响应完全结束
- 预热请求数：3
- 并发级别：1、2、4、8、16
- 每个并发级别的请求数：32
- 输出方式：流式输出
- temperature：0
- 最大输出长度（max_tokens）：64

每个并发级别分别发送 32 个请求。使用信号量（Semaphore）控制同一时刻正在执行的最大请求数量。

### 实验结果

| Concurrency | QPS | Output TPS | TTFT P95 (ms) | E2E P95 (ms) |
|---:|---:|---:|---:|---:|
| 1  | 5.22  | 334.0  | 11.3 | 197.0 |
| 2  | 9.69  | 618.9  | 16.6 | 212.6 |
| 4  | 19.31 | 1234.5 | 25.1 | 216.2 |
| 8  | 36.67 | 2346.9 | 30.4 | 225.5 |
| 16 | 68.03 | 4353.7 | 42.0 | 238.9 |

其中：

- QPS：服务每秒完成的请求数量。
- Output TPS：服务每秒生成的输出 token 总数。
- TTFT P95：95% 请求的首 token 延迟不超过该值。
- E2E P95：95% 请求的端到端延迟不超过该值。

### 实验结论

- 并发数从 1 增加到 16 时，QPS 和 Output TPS 持续增长。
- 并发数增加后，TTFT P95 和 E2E P95 也逐渐上升。
- 在并发数不超过 16 时，吞吐量没有出现平台或下降。
- 因此，本轮实验尚未达到服务的饱和点，需要继续提高并发数。

### 局限性

- 每个并发级别只测试了 32 个请求，样本数量较少。
- 本轮实验只运行了一次，没有验证结果的重复性。
- 最大并发数只有 16，未覆盖系统的饱和区间。
- 请求内容和输出长度固定，不能代表所有实际业务负载。

### 结果文件

- `concurrency_requests_httpx.csv`
- `concurrency_summary_httpx.csv`
- `concurrency_httpx_run.log`


## 3. 粗粒度饱和点实验

### 实验目的

进一步提高并发数，寻找系统吞吐量的峰值、性能拐点（knee point）以及开始过载的并发区间。

### 实验方法

- 测试脚本：`benchmark_saturation.py`
- 并发级别：16、32、64、128
- 每个并发级别的请求数：256
- 完整实验重复次数：3
- 预热请求数：3
- 输出方式：流式输出
- temperature：0
- 最大输出长度（max_tokens）：64
- HTTP 连接池容量：与最高并发数保持一致

增加每个并发级别的请求数量并重复三轮，用于降低偶然波动对结论的影响。

### 三轮平均结果

| Concurrency | QPS | Output TPS | TTFT P95 (ms) | E2E P95 (ms) |
|---:|---:|---:|---:|---:|
| 16  | 72.33  | 4624.6  | 29.3  | 227.2 |
| 32  | 127.61 | 8161.8  | 48.3  | 266.9 |
| 64  | 194.47 | 12443.3 | 112.4 | 373.8 |
| 128 | 144.00 | 9215.8  | 612.3 | 910.2 |

### 实验现象

- 三轮实验中，并发数 64 均取得了最高吞吐量。
- 并发数从 64 增加到 128 后，平均 Output TPS 下降约 26%。
- TTFT P95 从约 112.4 ms 增加到约 612.3 ms。
- E2E P95 从约 373.8 ms 增加到约 910.2 ms。
- 并发数 128 时，吞吐量下降且尾部延迟明显上升，说明请求开始大量排队。

### 实验结论

- 当前配置下，已测试到的最高吞吐量出现在并发数 64 附近。
- 系统在并发数 128 时已经进入明显的过载状态。
- 系统的饱和边界位于并发数 64 到 128 之间。
- 性能拐点大致位于并发数 32 到 64 之间。
- 并发数 32 更偏向较低延迟，並发数 64 更偏向最大吞吐量。
- 饱和点和性能拐点不是完全相同的概念：性能拐点通常早于完全饱和。

### 局限性

- 并发级别跨度较大，尚不能确定更加精确的性能拐点。
- 当前结果只适用于本机硬件、模型和 vLLM 配置。
- 使用固定 Prompt 和固定输出长度，负载类型比较单一。
- 使用本机回环网络，不包含实际生产环境的网络开销。
- GPU-Util 接近 100% 不等于服务一定已经达到吞吐量饱和。

### 结果文件

- `concurrency_requests_saturation_01.csv`
- `concurrency_summary_saturation_01.csv`
- `concurrency_requests_saturation_02.csv`
- `concurrency_summary_saturation_02.csv`
- `concurrency_requests_saturation_03.csv`
- `concurrency_summary_saturation_03.csv`
- `saturation_run_01.log`
- `saturation_run_02.log`
- `saturation_run_03.log`



## 4. 细粒度饱和点扫描实验

### 实验目的

在粗粒度实验确定的 C=64～128 区间内进行细粒度扫描，更准确地寻找最大吞吐量点、性能拐点和过载边界。

### 实验方法

- 测试脚本：`benchmark_saturation_fine.py`
- 并发级别：32、48、64、80、96、112、128
- 每个并发级别请求数：256
- 完整实验重复次数：3
- 预热请求数：3
- HTTP 客户端：HTTPX
- 输出方式：流式输出
- HTTP 连接池上限：128

### 三轮平均结果

| Concurrency | QPS | Output TPS | TTFT P95 (ms) | E2E P95 (ms) |
|---:|---:|---:|---:|---:|
| 32  | 126.53 | 8092.2  | 46.4   | 266.9  |
| 48  | 160.50 | 10270.8 | 78.7   | 317.1  |
| 64  | 188.59 | 12067.9 | 151.1  | 444.2  |
| 80  | 177.03 | 11325.8 | 277.8  | 647.3  |
| 96  | 126.91 | 8119.7  | 1184.7 | 1465.7 |
| 112 | 96.83  | 6197.3  | 2106.2 | 2363.3 |
| 128 | 135.60 | 8678.2  | 1084.0 | 1434.7 |

### 实验结论

- 三轮实验的最大吞吐量均出现在 C=64。
- 最大吞吐量约为 12068 output token/s。
- C=48～64 是性能拐点区域。
- C=64～80 是吞吐量饱和边界。
- 从 C=80 开始，吞吐量下降，同时尾部延迟快速上升。
- C>=96 时系统进入严重过载状态。
- 如果优先考虑延迟，可选择 C=48。
- 如果优先考虑最大吞吐量，可选择 C=64。

### 结果波动说明

过载区中的结果并不严格单调。例如 C=128 的部分结果好于 C=112。这可能受到动态批处理、请求排队和固定请求总数的影响，不代表系统性能在 C=128 时恢复。

判断过载不能只比较相邻两个并发点，而应同时观察多轮实验中的吞吐量趋势和尾部延迟。

### 局限性

- 每个并发级别固定为256个请求，高并发下稳定运行时间相对较短。
- 使用固定Prompt和固定输出长度，工作负载类型较单一。
- 使用0.5B小模型，结果不能直接推广到更大的模型。
- 测试使用本机回环网络，不包含远程网络延迟。
- 结论只适用于当前硬件、模型和vLLM配置。

### 结果文件

- `concurrency_summary_saturation_fine_01.csv`
- `concurrency_summary_saturation_fine_02.csv`
- `concurrency_summary_saturation_fine_03.csv`
- `concurrency_requests_saturation_fine_01.csv`
- `concurrency_requests_saturation_fine_02.csv`
- `concurrency_requests_saturation_fine_03.csv`
- `saturation_fine_run_01.log`
- `saturation_fine_run_02.log`
- `saturation_fine_run_03.log`





## 5. 输出长度对Decode性能的影响

### 实验目的

固定输入长度和并发数，只改变输出长度，观察输出长度对TTFT、TPOT、E2E、QPS和Output TPS的影响。

### 实验方法

- 压测工具：`vllm bench serve`
- 通信接口：OpenAI-compatible Chat Completions API
- 响应方式：SSE流式响应
- 随机内容长度：128 tokens
- 服务端实际Prompt长度：157 tokens/request
- 输出长度：16、32、64、128、256、512 tokens
- 并发数：1
- 每组正式请求：20
- 每组预热请求：3
- 每种长度重复次数：3
- `ignore_eos`：开启
- temperature：0

### 三轮平均结果

| Output Length | QPS | Output TPS | TTFT Mean (ms) | TPOT Mean (ms) | E2E Mean (ms) |
|---:|---:|---:|---:|---:|---:|
| 16  | 19.07 | 305.13 | 9.57 | 2.85 | 52.34 |
| 32  | 10.00 | 320.11 | 9.14 | 2.93 | 99.88 |
| 64  | 5.19  | 332.27 | 9.34 | 2.91 | 192.49 |
| 128 | 2.64  | 338.45 | 9.21 | 2.90 | 378.07 |
| 256 | 1.33  | 340.45 | 9.19 | 2.91 | 751.81 |
| 512 | 0.67  | 340.52 | 9.17 | 2.92 | 1503.53 |

### 实验结论

- 输出长度基本不影响TTFT。
- 单请求TPOT稳定在约2.9 ms/token。
- E2E随输出长度近似线性增加。
- 输出长度翻倍时，QPS大致减半。
- 短输出受到固定请求开销影响，Output TPS相对较低。
- 输出长度达到128后，Output TPS稳定在约340 token/s。
- 三轮结果波动较小，实验具有较好的重复性。

### 局限性

- 当前实验只测试了并发数1。
- 使用随机token输入，不代表真实对话内容。
- 结果只适用于当前模型、硬件和vLLM配置。
- 输入长度固定，尚未测量Prefill随输入长度变化的规律。

### 结果文件

- `output_length_runs.csv`
- `output_length_summary.csv`
- `output_length_run_01.log`
- `output_length_run_02.log`
- `output_length_run_03.log`
- `output_length_results/`



## 6. 输入长度对 Prefill 性能的影响

### 实验目的

固定输出长度和并发数，改变输入长度，观察 Prefill 阶段对 TTFT、端到端延迟和请求吞吐量的影响。

### 实验配置

- 日期：2026-09-07 至 2026-09-08
- 模型：Qwen/Qwen2.5-0.5B-Instruct
- 服务框架：vLLM
- 接口：OpenAI Chat Completions，SSE流式响应
- 设置输入长度：32、64、128、256、512、1024、2048 tokens
- 固定输出长度：16 tokens
- 并发数：1
- 每个输入长度正式请求数：20
- 每轮预热请求数：3
- 重复轮数：3
- random range ratio：0
- temperature：0
- ignore EOS：启用
- 网络：本机回环网络

在当前模型的聊天模板下，实际输入长度比设置输入长度固定多29 tokens。

### 三轮平均结果

| 设置输入 | 实际输入 | QPS | TTFT Mean | TTFT P95 | TPOT Mean | E2E Mean | E2E P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 61 | 19.09 | 9.82 ms | 12.14 ms | 2.83 ms | 52.32 ms | 54.65 ms |
| 64 | 93 | 18.59 | 9.67 ms | 11.79 ms | 2.95 ms | 53.88 ms | 59.29 ms |
| 128 | 157 | 18.69 | 9.73 ms | 11.14 ms | 2.91 ms | 53.42 ms | 55.04 ms |
| 256 | 285 | 17.95 | 11.89 ms | 12.71 ms | 2.92 ms | 55.63 ms | 56.87 ms |
| 512 | 541 | 16.16 | 16.71 ms | 19.68 ms | 3.01 ms | 61.84 ms | 67.44 ms |
| 1024 | 1053 | 15.01 | 23.05 ms | 25.23 ms | 2.90 ms | 66.57 ms | 68.98 ms |
| 2048 | 2077 | 11.82 | 41.54 ms | 44.51 ms | 2.87 ms | 84.52 ms | 87.10 ms |

### 结果分析

- 实际输入长度为61至157 tokens时，TTFT基本稳定在9.7至9.8 ms，固定开销占主导。
- 从实际输入285 tokens开始，TTFT逐渐上升。
- 实际输入达到541 tokens后，Prefill计算开销明显增长。
- 实际输入2077 tokens时，TTFT Mean达到41.54 ms，约为短输入时的4.2倍。
- TPOT始终稳定在约2.8至3.0 ms/token，说明输入长度主要影响 Prefill，对 Decode速度影响较小。
- 随着输入长度增加，QPS由19.09下降至11.82，E2E Mean由52.32 ms增加至84.52 ms。
- 512、1024和2048输入下的 TTFT P95轮间标准差分别为0.41、1.18和1.65 ms，修正后重复性良好。

### 异常与修正

第二轮曾被重复运行一次。由于相同 seed 生成了相同请求，第二次运行命中了 vLLM Prefix Cache，使长输入的 TTFT异常降低。

异常结果已保存到：

- `input_length_results/cache_hit_run_02/`
- `input_length_run_02_cache_hit.log`

随后重启 vLLM 清空缓存，并使用相同配置重新运行第二轮。修正后的第二轮与第一、三轮结果一致，正式汇总数据不包含缓存命中结果。

### 局限性

- 使用合成随机 Token，不能完全代表真实对话文本。
- 当前实验只测试单并发，未包含排队和连续批处理影响。
- 结果仅适用于当前GPU、模型、聊天模板和 vLLM配置。
- 本实验研究无跨轮缓存复用的 Prefill基线，不代表启用缓存复用后的线上性能。

### 结果文件

- `input_length_results/input_*_run_*.json`
- `input_length_runs.csv`
- `input_length_summary.csv`
- `input_length_summary.log`
- `summarize_input_length.py`





## 7. Prefix Cache 对照实验

### 实验目的

验证 vLLM Prefix Cache 是否能够降低重复输入的 Prefill 延迟，并排除重复运行本身带来的性能变化。

### 实验配置

- 原始输入长度：2048 tokens
- 实际输入长度：2077 tokens
- 输出长度：16 tokens
- 并发数：1
- 每组请求数：20
- 预热请求数：0
- 随机种子：7001
- 请求内容：两次运行完全相同
- 对照变量：Prefix Cache 开启与关闭

### 实验结果

| Prefix Cache | 请求状态 | QPS | TTFT Mean (ms) | TTFT P95 (ms) | E2E Mean (ms) |
|---|---|---:|---:|---:|---:|
| 开启 | Cold | 11.25 | 47.16 | 58.68 | 88.75 |
| 开启 | Warm | 18.09 | 12.62 | 13.88 | 55.16 |
| 关闭 | First | 11.52 | 43.72 | 46.48 | 86.72 |
| 关闭 | Repeat | 11.34 | 42.43 | 44.81 | 88.05 |

### 结果分析

开启 Prefix Cache 后，重复请求的 TTFT Mean 从 47.16 ms 降至 12.62 ms，降低约 73.2%；TTFT P95 降低约 76.3%；E2E Mean 降低约 37.8%；QPS 提高约 60.8%。

关闭 Prefix Cache 后，第一次与重复运行的 TTFT Mean 仅相差约 3%，QPS 和 E2E 也只有小幅波动，属于正常测量误差，没有出现重复请求加速。

### 结论

Prefix Cache 能够复用相同输入前缀对应的 KV Cache，显著减少重复请求的 Prefill 计算量，从而降低 TTFT 和端到端延迟并提高请求吞吐量。它主要优化输入处理阶段，不会直接提高 Decode 阶段的单 token 生成速度。
