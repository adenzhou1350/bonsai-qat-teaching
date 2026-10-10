# Bonsai QAT Teaching

### 从 CPU 上的三值实验，走到大模型训练与单卡压缩推理

**权重变成 `−1 / 0 / +1` 之后，梯度怎么传、训练怎么恢复、模型还能不能用？** 这个仓库用可读的 PyTorch 代码和实际实验记录，把这些问题逐步拆开。

[教学专题](https://adenzhou1350.github.io/learn/bonsai-low-bit/) · [CPU 起步](#cpu-start) · [动手课程](docs/LABS.md) · [方法说明](docs/METHOD.md) · [训练指南](docs/REPRODUCING.md) · [实验与评测](docs/experiments/README.md)

从一个不需要下载模型的小实验开始，阅读 **Qwen3.5-35B-A3B 全专家权重 QAT**，再研究 **Qwen3.5-122B-A10B 在单张 32GB RTX 5090 上的压缩推理**。你会接触三值量化、知识蒸馏、优化器检查点、打包格式，以及比“跑起来”更重要的数值与能力验证。

这是 **Bonsai 风格的教学与研究实现**。35B 和 122B 都是 MoE，总参数与每 token 激活参数不同；当前尚未建立完整 Bonsai 配方复现或质量无损的结论。仓库公开代码与实验记录，不包含模型权重。

## 当前结果：122B MoE，单张 32GB RTX 5090

已打包的 **Qwen3.5-122B-A10B** 文本模型已完成训练、导出、独立数值检查和单卡推理；这里的 122B 是总参数，每 token 激活约 10B。部署使用自定义三值/4bit 格式与 rank-8 补偿，全部文本权重驻留 GPU，无 CPU 权重卸载。

| 短请求部署口径 | 已验证结果 |
|---|---:|
| 权重与持久缓冲区 CUDA 存储 | 28.10 GiB |
| 全程已采样 NVML used 峰值 | 30.64 GiB |
| 驱动 reserved，另计 | 约 0.484 GiB |
| 最少已观察 free | 740 MiB |
| HTTP 原预填充 → 可选分组预填充，4 路合计输出吞吐 | **21.01 → 23.30 tokens/s（+10.90%）** |

HTTP 四轮采用原版/分组/分组/原版，各启动新服务，先热身 8 条再测 64 条，4 个闭环客户端；1536 总上下文、896MiB 缓存、输入≤128 token、输出上限64。吞吐包含 HTTP、调度、预填充与解码，排除加载、启动录图和显式热身。**这是四路合计吞吐，不能当作每个用户的速度。**

288 条请求全部 HTTP200/DONE，全部 SSE 与 57,036 条显存采样已独立复算。正式请求中 **244/256 与旧 CLI 输出 ID 相同**；另 12 条在代码 docstring 的中文措辞上变化，均保留记录。该实验完成了吞吐测量，没有通过严格逐 token 一致性验收；见[四轮数据与限制](docs/experiments/2026-10-10-1145-HTTP-observed-ABBA.json)。当前证据不足以保证继续修改还能获得明显收益，本轮研究已暂停。

另外两种口径单独保留：短请求 **CLI** 四轮为21.72→24.00输出 tokens/s，7,520个输出ID相同；较长输入252–1404 token的四轮为114.23→146.95 **input tokens/s**，测的是预填充较重的工作负载。这些数字不与 HTTP 收益相加，见[CLI 记录](docs/experiments/2026-10-10-0816-prefill16-deployment.json)和[输入吞吐记录](docs/experiments/2026-10-10-0921-long-input-ABBA.json)。上述显存余量很小，不能据此承诺更长上下文或更多并发。

## 有指定权重：启动 HTTP 服务

先按[推理环境说明](experimental122/README.md)准备 Linux、已测试的 Torch `2.11.0+cu130` / vLLM `0.24.0` 和 **native4095 指定 packed 产物**。仓库不附权重；原始 BF16 模型不能直接替代该目录。默认保留原预填充，`--grouped-prefill` 启用上表测过的可选优化。以下命令从仓库根目录运行，输出目录须不存在：

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/serve_vllm.py \
  --model /path/to/native4095-packed --output http-original
```

在另一终端调用默认的本机接口：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-native122","messages":[{"role":"user","content":"用一句话解释量化。"}],"temperature":0,"max_tokens":64,"stream":true}'
```

服务在请求完成后的正常停服已验证；持续压测、请求进行中的停服和生产延迟分布尚未验收。输出层 FP32 的独立诊断提供[可执行复现命令](experimental122/README.md#复现输出层fp32诊断)，短样本72/72输出ID相同；它仍是可选实验，没有替换默认部署或获得完整吞吐对照结论。

## 可以从这里学到什么

| 你想回答的问题 | 对应的实现与材料 |
|---|---|
| 三值权重如何训练？为什么还能反向传播？ | 分组三值网格、FP32 主权重、identity STE：[quantization.py](quantization.py) |
| 教师与学生到底比较哪些位置？ | 全位置 KL 与隐藏状态蒸馏：[prepare_teacher.py](prepare_teacher.py)、[train.py](train.py) |
| 中断恢复为什么不能只存模型权重？ | 主权重、Adam 动量、RNG 与文件校验：[checkpoint.py](checkpoint.py) |
| 为什么训练显存远高于部署显存？ | 35B 的全专家训练状态与两卡分层：[METHOD.md](docs/METHOD.md) |
| 单卡装下 122B MoE 之后，还要验证什么？ | 三值专家、4bit 非专家、低秩补偿与独立数值检查：[122B 案例](docs/experiments/122B.md) |

适合会写基础 PyTorch、想理解低比特训练与推理工程的读者。**没有大显卡，也可以先完成 CPU 方程实验、阅读核心实现和实验复盘。**

[四节动手课程](docs/LABS.md)把这些入口串成练习，每课提供预期输出、验收问题和资源限制；可以从 CPU 编码开始，按资源逐步进入训练与推理。

<a id="cpu-start"></a>

## 第一次来：从 CPU 实验开始

这一入口不下载模型、不加载 35B/122B 权重，测试张量在 CPU 上计算。当前完整检查点测试使用 POSIX 目录同步，按 **Linux CPU 环境**准备；Windows 原生执行尚未验证。

在独立 Python 环境中，从仓库根目录运行：

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' python test_equations.py
```

安装会下载 PyTorch 等依赖。根目录固定 Torch/Transformers 版本；GPU 课程需要另外按[环境说明](docs/REPRODUCING.md#1-环境和资源)选择匹配的 CUDA wheel，默认 pip 安装不等于原实验环境。

成功时输出以 `PASS: ternary grid, identity STE, packing, ...` 开头，覆盖：

- 三值编码、尺度与打包/解包，包括非整组 padding。
- STE 与原生专家梯度，以及分块 KL 与完整计算的梯度对照。
- 保存再恢复后，下一次更新及 Adam 状态一致；损坏检查点被拒绝。

接着打开 [test_equations.py](test_equations.py)，对照 [quantization.py](quantization.py) 与 [METHOD.md](docs/METHOD.md) 阅读。CPU 通过证明这些小规模机制通过检查；大模型训练、部署和质量评测有各自的验证入口。

## 再选一条进阶路线

| 路线 | 资源与输入 | 复刻范围 | 入口 |
|---|---|---|---|
| **35B 全专家 QAT** | Linux、2×B300、原始模型、本地文本 | 准备 tokens/教师，运行控制、训练、导出与恢复；长程实验已暂停 | [训练指南](docs/REPRODUCING.md#35b全专家权重qat) |
| **122B 固定产物推理** | Linux、32GB RTX 5090、指定 packed 权重 | 单请求、Graph/静态对照及 HTTP 服务 | [运行说明](experimental122/README.md) |
| **122B 从原始模型复刻** | 原始 BF16 权重、大内存 Linux CPU；后续教师/恢复另需 GPU | 校准数据重建和完整 CPU 低比特初始化已验证；完整恢复链路未完成 | [数据课程](courses/122b/README.md) · [初始化课程](courses/122b/seed_cpu/README.md) |

35B 更新全部专家 FP32 主权重；122B 案例冻结低比特基座，只训练 rank-8 补偿。两者训练成本和成绩不能互换；本仓库采用已验证的两张 B300 训练路线。

35B 专家训练状态约480GiB，两步控制实测每卡峰值约247.5GiB；完整优化器检查点约360GiB/份。磁盘、教师与主机内存要求见训练指南。根目录35B导出不能直接传给122B推理入口。

## 实验状态与质量

截至 **2026-10-10**，已完成的实验保留为教学案例；新的部署优化、两组 B300 长程训练及其等待任务均已停止，显卡已释放、已有检查点保留。后续恢复需重新发起任务，见[暂停与检查点记录](docs/experiments/2026-10-10-1150-research-paused.json)。

| 实验 | 最终记录状态 |
|---|---|
| **T35：全专家 QAT** | 全40层两步控制通过；长程主线最后记录2680步、最新保存2560步；低学习率续训最后记录1115步、保存点1024步。两组均暂停，未完成4096步长程质量验收 |
| **R35-8192：rank-16 补偿** | 8192步、数值/推理和四组开发评测完成；质量门槛未通过 |
| **NR35：norm/router 恢复** | 4096步、导出、独立128条重载、缓存/Graph和四组开发评测全部完成；未证明整体质量改善 |
| **R122-native4095：rank-8 补偿** | 4096步、导出、独立数值、单卡推理及四组开发评测完成；四组分数略低于对应 BF16 |
| **P122：完整可移植训练链路** | 校准和初始化等环节已验证；原始模型到同一122B恢复产物的完整公开链路仍未完成，已暂停 |

| 公开开发评测 | 35B rank-16 | 35B + norm/router | 对应35B BF16 | 122B native4095 | 对应122B BF16 |
|---|---:|---:|---:|---:|---:|
| HumanEval /164 | 141 | 136 | 129 | 134 | 135 |
| 中文及格式 /48 | 48 | 47 | 30 | 46 | 48 |
| CMMLU /201 | 158 | 159 | 161 | 161 | 163 |
| GSM8K /200 | 190 | 189 | 192 | 189 | 190 |

HumanEval 的不支持样本保留在分母内：35B rank-16为1个，norm/router和122B各2个；历史另一个35B原模型对照为133/164。各列沿用对应开发协议，不能跨模型大小排名。见[35B 完整评测](docs/experiments/2026-10-09-2245.json)、[norm/router 最终评测](docs/experiments/2026-10-10-1145-NR35-development-complete.json)及[122B 训练与评测](experimental122/native4095-training-result.json)。

项目使用者接受当前122B量化效果用于继续部署实验；原定严格质量门槛未通过的事实保留，保留测试未打开。权重、精确数据采样与独立评分流水线尚未全部发布，**仅 clone 仓库不能重建同一122B权重或复制全部分数**。目前可以复刻核心机制、已公开的数据/初始化环节，以及在取得指定权重后的推理流程。

## 代码阅读地图

```text
prepare_data.py       本地 JSONL → 按文档划分的 TRAIN / VAL
prepare_teacher.py   原始 BF16 模型 → 与学生位置匹配的教师目标
quantization.py      三值网格、STE、五个三值编码打包成一字节
model.py             原生 Qwen3.5 文本层与两卡分层
train.py             全专家 QAT、验证、导出
checkpoint.py        完整训练状态保存、校验、恢复
test_equations.py    CPU 小规模方程与恢复检查
experimental122/     固定 122B 产物的推理入口与 runtime
docs/experiments/    按路线、权重身份、时间分开的实验记录
```

三值编码为 1.6bit/权重，实际存储还包括尺度、padding、保留权重与缓存。方法说明解释这些数字怎样计算；[推理复盘](docs/experiments/INFERENCE.md)保留舍入、CUDA 后端、缓存与 Graph 的失败和修复；[代码身份清单](docs/reproducibility-manifest.json)用于核对已验证的实现。早期训练记录保存在[原始归档](docs/archive/2026-10-09-original-readme.md)，按时间追加的部署进展移至[部署归档](docs/archive/2026-10-10-deployment-progress-readme.md)。当前状态见[实验索引](docs/experiments/README.md)；旧日期快照不改写。

## 一起完善这门课

欢迎提交可复现的问题、教学解释改进和小规模对照实验。提 [Issue](https://github.com/adenzhou1350/bonsai-qat-teaching/issues) 时，请带上 commit、环境版本、运行命令、期望/实际结果和最小复现；实验比较请补充模型/产物身份、数据划分与计时口径，不要上传私有数据、密钥或模型权重。

新增训练或推理路线先留下控制实验与失败记录，再提升为可运行课程。**数值一致、训练完成、能力达标、服务性能，分别用证据回答。**

更多 AI Infra 与 Agent 实践见 [Aden 的项目与教学网站](https://adenzhou1350.github.io/)。

## English overview

A code-first learning repository for ternary quantization, straight-through gradients, knowledge distillation, checkpoint recovery, and compressed MoE inference. Start with small Linux CPU checks; explore full-expert QAT for Qwen3.5-35B-A3B on 2×B300 and a separate packed Qwen3.5-122B-A10B inference experiment on one 32GB RTX 5090.

This is a Bonsai-style research implementation, not a complete Bonsai reproduction. Weights are not included, the end-to-end 122B training recipe is incomplete, and the current 122B candidate falls below its matched BF16 baseline on all four reported development panels. Research is paused; completed results and checkpoints are preserved. See the [method](docs/METHOD.md), [reproduction guide](docs/REPRODUCING.md), and [dated evidence](docs/experiments/README.md).
