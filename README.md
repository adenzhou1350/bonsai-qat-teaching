<a id="从-cpu-上的三值实验走到大模型训练与单卡压缩推理"></a>

# Bonsai QAT Teaching

从 CPU 小实验开始，学习三值量化、蒸馏、检查点恢复和大模型压缩推理。仓库包含 PyTorch 教学代码、动手课程，以及实际实验的结果和失败记录。

[CPU 起步](#cpu-start) · [动手课程](docs/LABS.md) · [结果说明](docs/RESULTS.md) · [训练指南](docs/REPRODUCING.md) · [122B 推理指南](experimental122/README.md) · [网站导读](https://adenzhou1350.github.io/learn/bonsai-low-bit/)

这是 Bonsai 风格的独立教学与研究实现，尚未完整复现官方训练配方。**仓库不包含模型权重。** 35B 和 122B 都是 MoE，分别指总参数量，不是同规模稠密模型。

<a id="可以从这里学到什么"></a>
<a id="再选一条进阶路线"></a>
<a id="有指定权重启动-http-服务"></a>

## 先选适合你的入口

| 你想做什么 | 需要准备什么 | 从哪里开始 |
|---|---|---|
| 看懂三值编码、梯度和训练恢复 | Linux CPU、Python 与 PyTorch，无需模型 | [下方 CPU 实验](#cpu-start) → [第一课](docs/LABS.md) |
| 用自己的文本做 35B 全专家 QAT | Linux、2×B300、原始模型、数据和足够磁盘 | [训练指南](docs/REPRODUCING.md) |
| 运行已有 122B 压缩产物 | Linux、32GB RTX 5090、另行取得指定 packed 权重 | [推理与 HTTP 服务](experimental122/README.md) |
| 从原始 122B 权重学习数据校准与初始化 | 原始 BF16 模型、大内存 Linux CPU | [数据课程](courses/122b/README.md) → [初始化课程](courses/122b/seed_cpu/README.md) |

最后一条目前覆盖数据与初始化环节，完整教师生成、恢复训练与新产物部署尚未连通。原始 BF16 权重不能直接替代推理入口要求的 packed 权重。

<a id="当前结果122b-moe单张-32gb-rtx-5090"></a>
<a id="实验状态与质量"></a>

## 目前做到哪里

截至 **2026-10-10**，本轮研究已暂停，已有代码、结果和检查点保留。

- **基础教学**：三值编码、STE、蒸馏梯度、检查点恢复有小规模 CPU 检查；35B 整模型两步训练与导出已验证。
- **35B 全专家 QAT**：长程训练未完成，暂无完整质量验收。其他 35B 补偿实验的分数不属于这条路线。
- **122B 固定产物**：补偿训练、独立数值检查、单卡推理和 HTTP 服务实验已完成。四组开发评测仍低于对应 BF16 对照，原定质量门槛未通过。

最近的 HTTP 实验中，原预填充与可选分组预填充的**四客户端合计输出吞吐**分别为 **21.01 和 23.30 tokens/s**，已采样 NVML used 峰值 **30.64 GiB**，驱动另保留约 **0.484 GiB**。这是固定短请求的观测结果；256 条正式请求里有 12 条与旧 CLI 的输出 token 序列不同，原版与优化版都出现了差异。默认预填充没有因此替换。

[结果说明](docs/RESULTS.md)把质量、显存、HTTP 吞吐和 CLI 速度分开列出，解释每个分母与测试条件；[实验索引](docs/experiments/README.md)链接完整证据。当前没有质量无损、生产稳定性或完整 122B 一键复现的结论。

<a id="cpu-start"></a>

## 第一次来：从 CPU 实验开始

这一入口不下载模型，也不加载 35B/122B 权重。完整检查点测试使用 POSIX 目录同步，按 **Linux CPU 环境**准备；Windows 原生执行尚未验证。

在独立 Python 环境中运行：

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' python test_equations.py
```

成功时输出以 `PASS: ternary grid, identity STE, packing, ...` 开头，检查三值尺度与打包、STE 与蒸馏梯度，以及保存恢复后的下一次更新和 Adam 状态。损坏检查点会被拒绝。

安装会下载 PyTorch 等依赖。根目录固定了 Torch/Transformers 版本；GPU 课程还需匹配 CUDA wheel，详见[环境说明](docs/REPRODUCING.md#1-环境和资源)。CPU 测试通过只说明这些小规模机制通过检查。

## 读代码时，先分清训练路线

| 路线 | 哪些参数在训练 | 公开内容 |
|---|---|---|
| T35：35B 全专家 QAT | 全部专家的 FP32 主权重，前向时投影到三值 | 根目录训练、导出和完整优化器恢复代码 |
| R35 / NR35：35B 补偿与恢复 | 冻结低比特基座，训练 rank-16 补偿；NR35 再训练 norm/router | 实验记录，独立训练入口尚未发布 |
| R122：122B 补偿 | 冻结三值/4bit 基座，训练 rank-8 补偿 | 固定产物的推理代码和实验记录；可移植训练链路仍不完整 |

35B 全专家路线的专家训练状态约 480 GiB，两步控制实测每卡峰值约 247.5 GiB；单份完整检查点约 360 GiB。部署显存不能用来估算这条路线的训练资源。算式和实现区别见[方法说明](docs/METHOD.md)。

<a id="代码阅读地图"></a>

## 文件怎么读

| 文件或目录 | 内容 |
|---|---|
| [quantization.py](quantization.py) | 三值网格、STE、打包与解包 |
| [prepare_data.py](prepare_data.py) / [prepare_teacher.py](prepare_teacher.py) | 本地数据划分与位置匹配的教师目标 |
| [model.py](model.py) / [train.py](train.py) | 35B 模型、两卡分层、QAT 与导出 |
| [checkpoint.py](checkpoint.py) / [test_equations.py](test_equations.py) | 完整训练状态与 CPU 检查 |
| [experimental122/](experimental122/README.md) | 指定产物的 CLI、HTTP 与诊断入口 |
| [courses/122b/](courses/122b/README.md) | 数据校准与 CPU 初始化课程 |
| [docs/experiments/](docs/experiments/README.md) | 各条路线、原始 JSON 和历史时间线 |

早期配置、失败和中途进度保留在[推理历史](experimental122/HISTORY.md)与[实验时间线](docs/experiments/TIMELINE.md)。首次使用按当前指南选择入口；核对代码身份时使用[实现清单](docs/reproducibility-manifest.json)。

<a id="一起完善这门课"></a>

## 反馈与贡献

欢迎提交复现问题、教学解释改进和小规模对照实验。提 [Issue](https://github.com/adenzhou1350/bonsai-qat-teaching/issues) 时附上 commit、环境、命令、预期与实际结果；比较实验时再补充模型/产物、数据划分与计时范围。

更多项目与教程见 [Aden 的网站](https://adenzhou1350.github.io/)。

## English overview

A PyTorch teaching repository for ternary quantization, straight-through gradients, distillation, checkpoint recovery, and compressed MoE inference. Start with small Linux CPU checks. Full-expert 35B QAT and frozen-base low-rank recovery are separate experiments.

The fixed 122B MoE artifact runs on one 32GB RTX 5090, but weights are not included and the portable end-to-end training chain is incomplete. All four reported 122B development panels remain below their matched BF16 baselines. Research is paused; completed results and checkpoints are preserved. See [results and measurement scope](docs/RESULTS.md) before interpreting speed or quality numbers.
