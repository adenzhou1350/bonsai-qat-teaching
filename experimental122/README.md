# 单张 RTX 5090 跑 122B：已验证的实验版

本目录保存 **Qwen3.5-122B-A10B** 的新旧两版自定义量化推理入口：48 层三值专家、非专家线性层与 embedding 4bit、rank-8 BF16 补偿。总参数约 122B，每 token 激活约 10B。这不是 122B 稠密模型，也不是完整 Bonsai 复现。

**旧版已完成量化、4096 步补偿训练与打包，能在单张 32GB RTX 5090 上生成。质量仍未通过验收：中文/格式 45/48，验收线 48；CMMLU 162/201，验收线 163。** 新的教师对齐产物 `native4095` 也已完成训练、导出回放、独立数值检查，以及直接使用本仓库入口的单卡测速和中文生成；结果分别记录。新版质量也未通过验收。

## 新一轮训练记录（2026-10-09）

新版本在单张 B300 上完成 4096 步、16,773,120 个监督位置。训练的是全部 48 层的 192 个 FP32 rank-8 补偿张量；三值专家、校准后的非专家 4bit 权重保持冻结，**不是全专家权重 QAT**。使用 OpenThoughts 的 4096 条训练块和 128 条验证块，每块 4096 token；原始 BF16 教师与学生均监督 4095 个位置，教师使用原始输出头，学生使用自己的 4bit 输出头。

128 条验证样本的平均 `KL + 0.1 × normalized-state-MSE` 从 0.16703349 降至 0.04561734，128/128 条均改善。全部 48 层已导出，重新加载 BF16 补偿后，128 条验证结果与导出前完全一致，训练进程正常退出。机器记录见 [native4095-training-result.json](native4095-training-result.json)。

截至 2026-10-09 19:08（北京时间），独立 B300 进程重新加载后，全部 128 条验证、524,160 个监督位置的结果与训练记录完全一致；另完成 768 个逐层和 16 个整模型缓存检查。单张 RTX 5090 上也已完成 768+16 个缓存检查，隐藏状态与缓存 logits 的相对误差均为 0，16/16 个缓存 argmax 相同；该次 512-token 预填充加 1-token 解码检查的峰值 CUDA allocated 为 30,498,338,816 字节。

新版四组公开开发评测和逐题原模型对照已全部完成：

| 评测 | 新版通过数 | 原生 BF16 通过数 | 总题数 |
|---|---:|---:|---:|
| 中文及格式 | 46 | 48 | 48 |
| CMMLU | 161 | 163 | 201 |
| GSM8K | 189 | 190 | 200 |
| HumanEval | 134 | 135 | 164 |

**四组均未达到原模型基线，质量未通过验收。** 保持原数据、生成上限、EOS 和评分规则；两个不支持的 HumanEval 样本仍计入 164。最终测试流程已按既定开发评测条件关闭，没有读取保留测试数据。训练损失改善、数值一致和解码速度不能替代能力验收。完整汇总 SHA 和逐题结果分类计数见训练记录 JSON。

`infer.py` 默认保留旧版；使用 `--artifact-version native4095` 加载新版。两版均校验各自固定的 manifest SHA。新版本训练入口还在整理：除补偿训练脚本外，还需提供原始专家三值拟合、非专家 4bit 校准和匹配的教师目标生成流程，尚未发布可从原始模型开始的一键训练入口。

## 文件与运行环境

从 `infer.py` 读起；`runtime/` 只保留已验证推理路径需要的 25 个依赖模块，移除了训练、集群调度与评测入口。数值计算语句与原实验 AST 一致，来源记录在 `runtime-provenance.json`。保留这些模块是为了维持舍入、路由、缓存与内存释放顺序。

环境为 Linux、Python 3.12、Torch 2.11.0+cu130、Transformers 5.12.1、Triton。还需使用原环境的 `causal_conv1d` CUDA 扩展；缺少或替换实际卷积后端会改变计算。本目录没有修改系统包或自动安装 CUDA 扩展。模块对部分 Transformers 源码带 SHA 校验，版本或代码不一致时会拒绝运行。

模型权重不放进 Git。需要另行提供**所选版本已经打包好的实验模型目录**，包含配置、tokenizer、`summary.json`、`retained.pt`、embedding 和全部专家/dense bank 文件。原始 Hugging Face BF16 权重不能直接传给这个入口。本仓库目前没有从原始 122B 权重训练出该产物的一键流程；122B 的最小训练代码仍待完整验证。

旧版（默认 `--artifact-version old`）manifest SHA256：

```text
5c9683297d71c1dbdd1c32a8a749aa1911c857fb9ef1a788dfe794230abd08a9
```

新版（`--artifact-version native4095`）manifest SHA256：

```text
dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1
```

加载前逐文件校验 SHA；不需要实验机器的原目录、SSH 或旧调度日志。这里只验证文件身份与推理，不把去掉内部调度依赖解释为质量验收。

```bash
# 在已安装上述依赖的 Linux GPU 环境中运行。
CUDA_VISIBLE_DEVICES=0 python experimental122/infer.py \
  --model models/qwen35-122b-experimental-packed \
  --prompt '用中文解释什么是模型量化。' --max-new-tokens 128 \
  --output outputs/122b-generation.json

# 与原记录相同的五种上下文、固定 64-token Graph/静态对照。
CUDA_VISIBLE_DEVICES=0 python experimental122/infer.py \
  --model models/qwen35-122b-experimental-packed --benchmark \
  --output outputs/122b-benchmark.json
```

生成入口按 EOS 停止。测速入口固定执行 64 步，即使遇到 EOS 也继续，以对齐原来的工程记录；它排除提示词处理与 Graph 录制。单请求文本推理，无 CPU 权重卸载；没有并发服务、vLLM 插件或视觉推理支持。

## 已有实测

| 上下文 token | 解码 tokens/s | 峰值 CUDA allocated 字节 |
|---|---:|---:|
| 128 | 28.44 | 30,486,048,256 |
| 512 | 28.34 | 30,530,485,248 |
| 4096 | 25.56 | 31,042,519,552 |
| 8192 | 22.72 | 31,369,777,664 |
| 16384 | 18.81 | 32,494,375,424 |

全部五种上下文的 Graph/静态 64-token 输出逐位一致。文字模型去重后的 CUDA 权重/缓冲存储为 30,171,940,608 字节，无 CPU 权重卸载；峰值包含缓存与 Graph 工作区。以上是固定输入工程测试，不能推导任意长上下文理解、服务吞吐或质量已经恢复。

另有 768 个逐层与 16 个整模型缓存数值检查通过。这些数值对照与速度不能替代能力评测；本实验版适合研究和演示。


## 新版 native4095：仓库入口重新验证

2026-10-09，直接使用待提交的 `infer.py` 与全部 25 个 runtime 模块，在另一张空闲 RTX 5090 上完成五档 Graph/静态对照与中文生成；两个命令均退出 0。该入口源码与实测记录的 SHA 对应，数值装配路径保持原实验计算。记录见 [native4095-verified-run.json](native4095-verified-run.json)。

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/infer.py \
  --model models/qwen35-122b-native4095-packed --artifact-version native4095 \
  --prompt '用中文解释什么是模型量化。' --max-new-tokens 128 \
  --output outputs/122b-native4095-generation.json

CUDA_VISIBLE_DEVICES=0 python experimental122/infer.py \
  --model models/qwen35-122b-native4095-packed --artifact-version native4095 \
  --benchmark --output outputs/122b-native4095-benchmark.json
```

| 上下文 token | 解码 tokens/s | 峰值 CUDA allocated 字节 |
|---|---:|---:|
| 128 | 28.75 | 30,486,048,256 |
| 512 | 28.65 | 30,530,485,248 |
| 4096 | 25.84 | 31,042,519,552 |
| 8192 | 22.97 | 31,369,777,664 |
| 16384 | 19.01 | 32,494,375,424 |

五档均完成固定 64-token Graph/静态 token ID 一致检查。去重后 CUDA 文本权重及缓冲存储为 30,171,940,608 字节，无 CPU 权重卸载。速度只计解码，排除预填充与 Graph 录制，每档一个计时样本；中文控制只生成最多 32 token，用于确认入口执行。以上没有验证并发服务或长上下文能力，质量失败仍保留。

## 常驻模型与跨请求 Graph 复用

新增 [run_requests.py](run_requests.py) 一次加载 native4095，串行读取 JSONL 请求，复用同一个 CUDA Graph。原 `infer.py` 与 25 个原始 runtime 模块保持原样；新增模块身份在 [request-reuse-provenance.json](request-reuse-provenance.json)。它不更改量化权重、路由或算子精度。

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/run_requests.py \
  --model models/qwen35-122b-native4095-packed \
  --requests experimental122/examples/deployment-requests.jsonl \
  --max-new-tokens 96 --max-cache-len 1536 \
  --output outputs/122b-resident-requests.json
```

每行格式为 `{"prompt":"你的问题"}`。请求按 EOS 或输出上限停止；每次重置缓存并预填充，首次录制 Graph，后续复用固定地址。输出记录首 token 延迟、完整请求耗时、显存及 token ID。选择新的输出文件；已存在的结果会被拒绝覆盖。

在三条短提示词、每种模式重复三次的相同权重对照中，首 token 中位数约 **6.06 → 1.52 秒**，包含准备时间的串行请求吞吐约 **10.21 → 18.99 tokens/s**；全部 2592 个输出 token ID 相同。独立新进程再次执行以上新入口，三条请求的 288 个输出 token 与旧路径相同，正常退出。详见[计时范围与完整记录](../docs/experiments/INFERENCE.md#122b部署优化与显存口径)。

这些请求都达到 96-token 上限，没有测到 EOS；不是完整回答质量成绩。模型加载不计入请求吞吐，纯解码仍约 28 tokens/s。固定缓存容量 1536 的短请求已验证；固定容量 16384 的复用路径也已通过另一组新进程对照，实际输入顺序为 103/3763/451/14755/7543/103，319 个输出 token 全部一致，其中三个请求到达 EOS。详见[长短混合请求完成记录](../docs/experiments/2026-10-09-2354.json)。该入口是单 worker 串行执行，不是多请求 batch 或并发 HTTP 服务。

固定 16K 缓存下，六次混合请求的整卡观测值最高约 **31.33GiB**，当时仅余约 34.3MiB；峰值 allocated 约 30.00GiB。它证明该组请求实际完成，不代表还有充足余量。固定大缓存也会让短请求纯解码降至约 18.8 tokens/s；短请求可先用已验证的默认 1536 容量。已发布下述可选缓存优化；原固定 16K 测量作为历史对照保留。

## 按请求选择缓存容量与释放空闲块

同一份新版 `run_requests.py` 已在三个独立 GPU 进程完成默认短请求、固定 16K 清理、自适应容量清理：共 21 个请求、1245 个输出 token ID 全部与原量化路径一致，均退出 0。连续两次 14755-token 输入及两次 7543-token 输入也已完成。见[当前代码的完成证据](../docs/experiments/2026-10-10-0014.json)。

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/run_requests.py \
  --model models/qwen35-122b-native4095-packed \
  --requests experimental122/examples/deployment-requests.jsonl \
  --max-new-tokens 64 --max-cache-len 16384 \
  --auto-cache-len --trim-prefill-cache \
  --output outputs/122b-adaptive-requests.json
```

以上命令的示例文件含三条短提示；长输入实测使用另一组自拟提示，实际 token 长度及容量顺序保存在完成证据中。使用自己的 JSONL 时，容量须严格大于实际聊天模板输入长度加生成上限和 4-token 余量。

`--auto-cache-len` 在 1536/4096/8192/16384 和配置上限中选择可用最小容量，只保留一份 Graph/缓存。容量切换或超过 4096-token 的长预填充前释放旧 Graph，随后重新录制；同容量短请求继续复用。`--trim-prefill-cache` 在预填充后归还未使用的分配器块。两个参数默认关闭；不修改权重、路由和 EOS。

新版默认短请求的整卡请求末观察最大约 **29.18GiB**；固定 16K 清理约 **29.83GiB**；自适应混合请求约 **30.03GiB**，其中短请求恢复到约 27.5 tokens/s 纯解码，长输入约 18.8 tokens/s。固定 16K 预填充时、清理前仍观察到约 **31.32GiB**，清理不能消除预填充峰值；这些边界观察不是连续采样峰值。自适应切换和长输入重新录制有时间成本，不保证所有请求组合都更快。

代码结果中的保守 `limits` 字段仍保留初版“其他容量需另行验证”的说明；新版已验证范围以上述独立完成证据为准。当前仍为单 worker 串行入口，固定长度 batch 的实验速度不代表这个 CLI 的服务吞吐。
