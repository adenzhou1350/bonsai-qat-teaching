# Bonsai 风格量化教学：从三值训练到单卡 122B 推理

用尽量少的代码理解三值量化、知识蒸馏、检查点恢复与压缩推理，并用真实实验说明每一步能证明什么。

主课程是 **Qwen3.5-35B-A3B 全专家权重 QAT，2×B300**；进阶案例是 **Qwen3.5-122B-A10B 三值专家 + 4bit 非专家 + rank-8 补偿，单张 RTX 5090 推理**。两者是 MoE，总参数和每 token 激活参数不同。本实现是 Bonsai 风格实验，尚未建立完整 Bonsai 训练方案复现或能力恢复的结论。

## 先选一条学习路线

| 目标 | 入口 | 需要的资源 | 仓库已验证到哪里 |
|---|---|---|---|
| 理解三值网格、STE、打包与恢复 | [方程测试](test_equations.py)、[方法说明](docs/METHOD.md) | CPU，已安装依赖 | 小规模方程、梯度、检查点下一步一致性通过 |
| 从原始 35B 模型训练专家权重 | [复刻步骤](docs/REPRODUCING.md#35b全专家权重qat) | Linux、2×B300、模型与本地文本 | 仓库入口全 40 层两步控制通过；4096 步长程仍在运行 |
| 在单张 5090 运行已打包 122B | [122B 运行说明](experimental122/README.md) | Linux、32GB RTX 5090、另行取得对应 packed 权重 | 新旧入口均已生成、测速；新版五档上下文约 19–29 tokens/s |
| 从原始 122B 复刻量化与补偿训练 | [122B 实验与缺口](docs/experiments/122B.md) | B300 训练与 CPU 校准资源 | 完整可移植链路仍在验证，尚未提供可执行的全流程课程 |

**122B 能运行，质量尚未达标。** 新版四组公开开发评测均略低于对应 BF16 原模型；损失下降、重载一致与测速成功不能替代能力验收。仓库不包含模型权重，当前不能仅凭克隆仓库重建 122B 产物。

## 课程顺序

[动手课程](docs/LABS.md) 把下面各步串成练习，并列出预期输出、验收问题和资源限制。

1. 读 [METHOD.md](docs/METHOD.md)：哪些参数训练、哪些冻结，1.6bit 编码怎样算。
2. 跑 `CUDA_VISIBLE_DEVICES='' python test_equations.py`，观察量化、梯度及恢复。
3. 按 [REPRODUCING.md](docs/REPRODUCING.md) 准备数据、教师目标，先两步再完整训练。
4. 读 [实验索引](docs/experiments/README.md)，比较全权重 QAT、低秩补偿与推理适配各自的证据。
5. 有对应 packed 权重时，再按 [experimental122](experimental122/README.md) 运行 5090 推理。

## 最小训练代码

| 文件 | 职责 |
|---|---|
| [quantization.py](quantization.py) | 分组三值网格、identity STE、五个三值打包成一字节 |
| [model.py](model.py) | 原生 Qwen3.5 文本层、完整层分配到两张卡 |
| [prepare_data.py](prepare_data.py) | 本地 JSONL → 按文档划分的 TRAIN/VAL token 块 |
| [prepare_teacher.py](prepare_teacher.py) | 原始 BF16 模型 → 与学生输入位置匹配的隐藏状态 |
| [train.py](train.py) | 全专家 FP32 主权重训练、验证与三值导出 |
| [checkpoint.py](checkpoint.py) | FP32 主权重、Adam 动量、RNG 保存与恢复 |
| [test_equations.py](test_equations.py) | CPU 方程、梯度、打包、恢复与损坏文件拒绝测试 |

35B 专家训练状态约 480GiB，实测两步峰值约 247.3/247.5GiB 每卡；完整检查点约 360GiB/份。部署显存和训练显存差别很大。此课程按已验证的 2×B300 路径编写，详细磁盘、数据和环境要求见复刻步骤。

## 实验记录怎么读

所有成绩按模型、训练路线、权重身份和时间分开记录：

- [35B 训练实验](docs/experiments/35B.md)：全专家 QAT、rank-16 补偿及第二轮训练。
- [122B 训练与质量](docs/experiments/122B.md)：已完成产物、质量失败、可移植全流程进度。
- [推理适配与优化](docs/experiments/INFERENCE.md)：舍入/后端差异、vLLM 桥接、Graph 和测速口径。
- [初始状态快照](docs/experiments/2026-10-09.json)与[后续完成节点](docs/experiments/README.md)：保留每个时点的真实状态，去除机器地址、PID 和凭证。
- [原 README 实验日志](docs/archive/2026-10-09-original-readme.md)：保留历史失败和修复，不再向首页追加流水账。

数值一致、训练完成、能力达标、服务性能是四种独立结论。新实验先记录真实状态，通过完整链路验证后再加入可运行课程；未验证脚本不会被描述成“一键复刻”。本次整理所对应的已验证代码身份见 [复刻清单](docs/reproducibility-manifest.json)。
