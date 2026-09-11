# Troubleshooting

## 1. OpenAI SDK 流式响应关闭异常

### 现象

使用 OpenAI Python SDK 进行并发流式压测时，实验完成后出现：

`RuntimeError: generator didn't stop after athrow()`

### 影响判断

虽然终端出现异常，但 160 个正式请求都成功写入 CSV：

- `concurrency_requests.csv` 共 161 行
- `concurrency_summary.csv` 共 6 行

因此这不是 vLLM 推理失败、CUDA 故障或显存不足。

### 定位过程

错误栈来自 `httpcore2` 的异步字节流清理过程。

推测 OpenAI SDK 收到 SSE 的 `data: [DONE]` 后开始结束流式响应，但底层 HTTP 响应可能还没有被完整消费。

### 对照方案

保留 `benchmark_concurrency_sdk.py` 作为 SDK 版本，新增 `benchmark_concurrency_httpx.py`，直接使用 `httpx` 读取 SSE。

HTTPX 版本收到 `[DONE]` 后不立即使用 `break` 退出，而是继续读取到底层 HTTP EOF，然后正常关闭响应。

### 验证结果

- 最小 SSE 客户端连续运行 10 次无异常
- HTTPX 并发实验完成 160 个请求
- 请求结果文件和汇总文件行数正确
- 日志中没有 RuntimeError、Traceback 或 athrow
- HTTPX 与 SDK 版本的吞吐量处于相近范围

### 结论

问题发生在客户端流式响应的清理路径中，不是模型推理失败。完整消费 HTTP 响应后，异常消失。






## 2. 重复压测导致 Prefix Cache 命中

### 问题现象

输入长度实验的第二轮被重复执行了一次。汇总后发现第二轮 TTFT 明显低于第一、三轮，而且输入越长，差异越明显。

以实际输入2077 tokens为例：

- 第一轮 TTFT Mean：41.78 ms
- 异常第二轮 TTFT Mean：14.26 ms
- 第三轮 TTFT Mean：40.38 ms

### 影响判断

第二轮没有请求失败或程序报错，但测量条件已经发生变化，因此不能作为无缓存 Prefill 基线参与平均。

### 定位过程

- 第二轮被重复运行。
- 两次运行使用相同 seed，因此生成了相同的随机请求。
- 第二次运行时，服务端可能复用了第一次留下的 Prefix Cache。
- 第二次运行使用相同文件名，覆盖了第一次生成的第二轮结果。

### 根本原因

重复向同一个 vLLM 服务发送相同 Prompt，使请求命中 Prefix Cache，减少了需要重新计算的 Prefill Token，从而导致 TTFT 异常降低。

### 修正方法

1. 将异常 JSON 和日志保存到独立目录。
2. 重启 vLLM，清除已有的 Prefix Cache。
3. 保持其他实验参数不变，重新运行第二轮。
4. 重新生成汇总结果并比较三轮数据。

修正后，实际输入2077 tokens时：

- 第一轮 TTFT Mean：41.78 ms
- 第二轮 TTFT Mean：42.45 ms
- 第三轮 TTFT Mean：40.38 ms

三轮结果恢复一致。

### 预防方法

- 每个独立轮次使用不同 seed。
- 不要在同一个服务进程中重复运行相同轮次。
- 必须重跑时，先重启服务或清空 Prefix Cache。
- 重跑实验时使用新的文件名，避免直接覆盖原始结果。
- 汇总前检查各轮数据，不能只检查请求是否成功。

### 保留文件

- `input_length_results/cache_hit_run_02/`
- `input_length_run_02_cache_hit.log`

### 结论

该问题不是推理失败，而是缓存导致的实验数据污染。压测成功不代表数据一定有效，还需要检查不同轮次之间是否具有一致的实验条件。




## 3. WSL 重启后环境变量未加载导致 vLLM 启动失败

### 问题现象

WSL 终端意外退出或重启后，原来的 vLLM 服务进程消失。重新启动 vLLM 时，API Server 无法正常启动，并出现：

`RuntimeError: UVA is not available`

以及：

`RuntimeError: Engine core initialization failed`

此时执行健康检查 `curl -i --max-time 3 http://localhost:8000/health`，也无法连接到 8000 端口。

### 影响判断

问题发生在 vLLM 服务启动阶段，模型尚未开始处理请求，因此不是压测代码错误，也不是推理过程中的显存不足。

已经生成并保存的 CSV、JSON 和日志文件不会因为服务进程退出而丢失。

### 定位过程

首先在 `server.log` 中搜索启动错误，发现最主要的错误是：

`RuntimeError: UVA is not available`

随后检查 CUDA 和编译器环境：

- `CUDA_HOME` 为空
- `CC` 为空
- `CXX` 为空
- `which nvcc` 没有返回预期路径
- 当前默认 GCC 不是项目之前使用的 GCC 14

进一步检查发现，CUDA 12.9、GCC 14 和 G++ 14 实际已经安装：

- `/usr/local/cuda-12.9/bin/nvcc`
- `/usr/bin/gcc-14`
- `/usr/bin/g++-14`

因此问题不是这些程序不存在，而是 WSL 重启后当前 Shell 没有恢复项目需要的环境变量。

### 根本原因

之前的 CUDA 路径、编译器路径和 WSL2 Pin Memory 配置只在原来的终端会话中生效，没有被持久化。

WSL 重启会结束原有进程，并清除临时 Shell 环境。重新打开终端后，vLLM 无法获得原来的 CUDA 和编译器配置，最终导致 Engine Core 初始化失败。

### 修正方法

将以下环境变量写入 `~/.bashrc`：

- `export VLLM_WSL2_ENABLE_PIN_MEMORY=1`
- `export CUDA_HOME=/usr/local/cuda-12.9`
- `export PATH="$CUDA_HOME/bin:$PATH"`
- `export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"`
- `export CC=/usr/bin/gcc-14`
- `export CXX=/usr/bin/g++-14`

执行 `source ~/.bashrc`，让配置立即在当前终端生效。

然后进入项目目录并激活虚拟环境：

- `cd ~/projects/llm-serve-bench`
- `source .venv/bin/activate`

重新执行原来的 vLLM 服务启动命令。

### 验证结果

服务端成功出现：

- `Application startup complete.`
- `API server: HTTP server started`

在另一个终端执行 `curl -i --max-time 3 http://localhost:8000/health`，返回：

`HTTP/1.1 200 OK`

说明 vLLM 服务已经恢复。

### 补充说明

WSL 启动时还出现了以下警告：

`wsl: 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。`

该警告与本次 vLLM 启动失败没有直接关系。本项目使用 WSL 内部的 `localhost:8000` 访问 vLLM，健康检查成功后可以暂时忽略该警告。

### 预防方法

- 将长期使用的环境变量写入 `~/.bashrc`，不要只执行临时的 `export`。
- 新开终端后确认项目虚拟环境已经激活。
- 启动服务前检查 `CUDA_HOME`、`CC`、`CXX` 和 `nvcc`。
- WSL 重启后需要重新启动 vLLM，原来的服务进程不会自动恢复。
- 先完成 `/health` 健康检查，再运行正式压测。

### 结论

本次故障由 WSL 重启后运行环境未正确恢复引起，不是模型推理、CUDA 硬件或压测程序本身发生故障。将环境配置持久化到 `~/.bashrc` 后，vLLM 可以正常启动。
