# Bonsai QAT Teaching

### 从 CPU 上的三值实验，走到大模型训练与单卡压缩推理

**权重变成 `−1 / 0 / +1` 之后，梯度怎么传、训练怎么恢复、模型还能不能用？** 这个仓库用可读的 PyTorch 代码和实际实验记录，把这些问题逐步拆开。

[教学专题](https://adenzhou1350.github.io/learn/bonsai-low-bit/) · [CPU 起步](#cpu-start) · [动手课程](docs/LABS.md) · [方法说明](docs/METHOD.md) · [训练指南](docs/REPRODUCING.md) · [实验与评测](docs/experiments/README.md)

从一个不需要下载模型的小实验开始，阅读 **Qwen3.5-35B-A3B 全专家权重 QAT**，再研究 **Qwen3.5-122B-A10B 在单张 32GB RTX 5090 上的压缩推理**。你会接触三值量化、知识蒸馏、优化器检查点、打包格式，以及比“跑起来”更重要的数值与能力验证。

这是 **Bonsai 风格的教学与研究实现**。35B 和 122B 都是 MoE，总参数与每 token 激活参数不同；当前尚未建立完整 Bonsai 配方复现或质量无损的结论。仓库公开代码与实验记录，不包含模型权重。

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

| 路线 | 资源与输入 | 你可以做什么 | 入口 |
|---|---|---|---|
| **35B 全专家 QAT** | Linux、2×B300、原始模型、本地文本 | 准备 tokens 和 BF16 教师，先两步控制，再训练、导出与恢复 | [训练指南](docs/REPRODUCING.md#35b全专家权重qat) |
| **122B 固定产物推理** | Linux、32GB RTX 5090、指定版本的 packed 权重 | 运行单请求文本生成与固定 64-token Graph/静态对照 | [运行说明](experimental122/README.md) |
| **122B 从原始模型复刻** | 校准、教师、训练与独立验证资源 | 目前先研究方法与各环节证据；完整可移植链路仍在整理 | [已验证环节与缺口](docs/experiments/122B.md) |

35B 的全专家 QAT 更新全部专家 FP32 主权重；122B 案例冻结低比特基座，只训练 rank-8 补偿。**它们是两条不同路线，训练成本与实验分数不能互换。**

35B 专家训练状态约 480GiB，两步控制实测峰值约 247.3/247.5GiB 每卡；完整优化器检查点约 360GiB/份。预留磁盘、环境与数据要求见训练指南。根目录 35B 导出不能直接传给 `experimental122/infer.py`。

## 已经做到了哪里

下表是 **2026-10-09 已提交记录**的摘要，进行中的实验以带时间的记录为准。

| 实验 | 已有证据 | 尚不能得出的结论 |
|---|---|---|
| **T35：35B 全专家 QAT** | 全 40 层两步控制、80 个专家梯度、完整检查点和导出核验通过 | 长程质量尚未完成；不能用补偿路线的成绩替代 |
| **R35-8192：35B rank-16 补偿** | **训练、数值/推理检查及四组完整开发评测已完成**，逐题汇总审计完成，见下表 | CMMLU、GSM8K 低于原模型，质量未通过；保留测试关闭 |
| **R122-native4095：122B rank-8 补偿** | 4096 步训练、独立数值检查、5090 五档推理对照完成 | 四组公开开发评测均低于对应 BF16 原模型，质量未通过 |

35B 的 8192 步产物已完成以下完整题集评测（2026-10-09 22:45 北京时间）：

| 35B 公开开发评测 | 状态 | R35-8192 | 对应 BF16 | 总题数 |
|---|---|---:|---:|---:|
| HumanEval | 已完成 | 141 | 129 | 164 |
| 中文及格式 | 已完成 | 48 | 30 | 48 |
| CMMLU | 已完成，未达基线 | 158 | 161 | 201 |
| GSM8K | 已完成，未达基线 | 190 | 192 | 200 |

HumanEval 本轮 1 个不支持样本仍计入 164 分母，另一个历史原模型对照为 133/164。评分规则、EOS、生成上限和完整题集保持不变。**四组评测与逐题汇总均已完成，质量未通过**，见[22:45 完整评测记录](docs/experiments/2026-10-09-2245.json)。保留测试的四条等待流程均已因开发门槛未通过而关闭，没有读取保留题。

122B 新版在 128/512/4K/8K/16K 上下文的已记录解码速度约 **19–29 tokens/s**，文本权重与缓冲全部驻留单张 RTX 5090，无 CPU 权重卸载。口径是每档一个样本、固定 64-token 解码，排除预填充与 Graph 录制；不代表端到端吞吐、并发服务或长上下文能力。

| 122B 公开开发评测 | native4095 | 对应 BF16 | 总题数 |
|---|---:|---:|---:|
| 中文及格式 | 46 | 48 | 48 |
| CMMLU | 161 | 163 | 201 |
| GSM8K | 189 | 190 | 200 |
| HumanEval | 134 | 135 | 164 |

两个不支持的 HumanEval 样本保留在 164 分母内。质量门槛未通过，保留测试保持关闭。模型权重、完整数据修订/采样清单及独立评分流水线尚未随仓库发布，因此目前不能仅凭 clone 重建 122B 产物或复制全部质量数字。

项目使用者现已接受 122B 当前量化效果，后续优先优化部署性能，历史质量成绩继续保留。新增[常驻请求入口](experimental122/README.md#常驻模型与跨请求-graph-复用)已通过独立新进程验证；短请求对照的首 token 中位数约 **6.06 → 1.52 秒**，包含准备的串行请求吞吐约 **10.21 → 18.99 tokens/s**，输出 token 全部相同。见[显存、计时范围与限制](docs/experiments/INFERENCE.md#122b部署优化与显存口径)。

显存须区分权重与运行峰值：122B 文本权重及缓冲 **28.10GiB**，短请求末整卡观察约 **29.18GiB**；长短混合优化版全程 NVML v2 采样最大 **30.755GiB used**，另列驱动 reserved **0.484GiB**，该时刻 free **618.3MiB**。单张 32GB 5090 已运行，但长预填充余量较小；采样可能遗漏更快瞬态。见[完整显存采样](docs/experiments/2026-10-10-0105.json)。

查看原始证据：[35B 记录](docs/experiments/35B.md) · [8192 步完成节点](docs/experiments/2026-10-09-2110.json) · [独立重载](docs/experiments/2026-10-09-2129.json) · [B300 缓存检查](docs/experiments/2026-10-09-2132.json) · [35B 5090 缓存与 Graph](docs/experiments/2026-10-09-2156.json) · [122B 训练与质量](experimental122/native4095-training-result.json) · [122B 5090 实测](experimental122/native4095-verified-run.json)

[初始状态快照](docs/experiments/2026-10-09.json)与[后续完成节点](docs/experiments/README.md)保留每个时点的真实状态，旧快照不改写成后来的结果；机器地址、PID 和凭证不写入公开记录。

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

三值编码为 1.6bit/权重，实际存储还包括尺度、padding、保留权重与缓存。方法说明解释这些数字怎样计算；[推理复盘](docs/experiments/INFERENCE.md)保留舍入、CUDA 后端、缓存与 Graph 的失败和修复；[代码身份清单](docs/reproducibility-manifest.json)用于核对已验证的实现。早期实验流水记录保存在[归档](docs/archive/2026-10-09-original-readme.md)。

## 一起完善这门课

欢迎提交可复现的问题、教学解释改进和小规模对照实验。提 [Issue](https://github.com/adenzhou1350/bonsai-qat-teaching/issues) 时，请带上 commit、环境版本、运行命令、期望/实际结果和最小复现；实验比较请补充模型/产物身份、数据划分与计时口径，不要上传私有数据、密钥或模型权重。

新增训练或推理路线先留下控制实验与失败记录，再提升为可运行课程。**数值一致、训练完成、能力达标、服务性能，分别用证据回答。**

更多 AI Infra 与 Agent 实践见 [Aden 的项目与教学网站](https://adenzhou1350.github.io/)。

## English overview

A code-first learning repository for ternary quantization, straight-through gradients, knowledge distillation, checkpoint recovery, and compressed MoE inference. Start with small Linux CPU checks; explore full-expert QAT for Qwen3.5-35B-A3B on 2×B300 and a separate packed Qwen3.5-122B-A10B inference experiment on one 32GB RTX 5090.

This is a Bonsai-style research implementation, not a complete Bonsai reproduction. Weights are not included, the end-to-end 122B training recipe is incomplete, and the current 122B candidate falls below its matched BF16 baseline on all four reported development panels. See the [method](docs/METHOD.md), [reproduction guide](docs/REPRODUCING.md), and [dated evidence](docs/experiments/README.md).

122B 显存实测：文本张量常驻约 **28.10GiB**；默认短缓存的请求末整卡观察最大约 **29.75GiB**。固定 16K 容量、最长 14755-token 输入的混合请求已验证输出一致，但整卡观察最高约 **31.33GiB**，余量很小。见[显存与长短请求对照](docs/experiments/2026-10-09-2354.json)。新版缓存优化已完成并发布：短请求整卡观察约 **29.18GiB**，固定 16K 清理后约 **29.83GiB**，连续长请求的自适应版约 **30.03GiB**；预填充瞬时仍接近卡容量。三个独立进程、1245 个输出 token 全部与原量化路径一致，见[完成证据](docs/experiments/2026-10-10-0014.json)。
