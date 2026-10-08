# Bonsai 风格三值 QAT：最小教学实现

把 Qwen3.5 MoE 的**完整专家权重**训练到 `{-1, 0, +1}` 网格。保留原生路由、SiLU、矩阵乘法和累加；注意力、嵌入、路由器、归一化和输出头保持原精度。没有新增输入归一化，也没有激活量化。

这是我们实验代码的教学整理，**不是 PrismML 私有 Bonsai 训练方案的完整复现**。首版聚焦 35B 全专家权重 QAT；正在实验的 122B「三值专家 + dense/embedding 4bit + rank-8 补偿」流程尚未合并进来。

## 从哪里读

| 文件 | 内容 |
|---|---|
| `quantization.py` | 分组三值网格、STE、5 个三值打包成 1 字节 |
| `model.py` | 加载 Qwen3.5 MoE 文本层、按整层分配到 GPU |
| `prepare_data.py` | 本地 JSONL 文本 → 独立训练/验证文档 → token 块 |
| `prepare_teacher.py` | 原始 BF16 模型 → 固定教师隐藏状态 |
| `train.py` | 全专家权重训练、验证、三值导出 |
| `checkpoint.py` | 逐权重张量保存完整 FP32 Adam 状态与恢复 |
| `test_equations.py` | 小规模方程、梯度、打包和检查点测试 |

## 量化与训练

每 128 个权重构成一组，8 次 Lloyd 更新估计尺度。最终尺度存 BF16，前向权重是 `BF16(code × scale)`。反向用 identity STE，把 BF16 网格的上游梯度传给 FP32 主权重；每次前向重新计算网格，三值编码和尺度都会随主权重变化。

训练全部专家的 FP32 主权重，AdamW 学习率默认 `1e-4`、无 weight decay、梯度范数裁剪到 1。损失是：

```text
KL(原模型输出分布 || 量化模型输出分布)
  + 0.1 × mean((student_hidden - teacher_hidden)^2)
          / mean(teacher_hidden^2)
```

KL 覆盖**所有 next-token 位置和整个词表**。`chunk=32` 是词表头按 32 个 token 分块重算以节省显存，不是只抽样 32 个位置。教师隐藏状态先写盘，训练时不用同时驻留原始教师模型。每条样本长度默认 4096，输入及监督位置数是 4095；4096 条训练块只遍历一次，约 1677 万监督位置。验证集不参与梯度或检查点选择。

三值编码的信息量下界是 `log2(3) ≈ 1.585 bit`。本实现 5 个三值/字节，编码约 1.6 bit/权重，另有每组 BF16 尺度和保留的高精度权重，**整模型不是恰好 1.58bit**。

## 硬件：不限定 B300，但 8×5090 不能直接跑这一版全权重训练

Qwen3.5-35B-A3B 有约 322 亿专家权重。本代码每个专家权重保存 FP32 主权重、FP32 梯度、两个 FP32 Adam 动量，共 16 字节：**仅专家训练状态约 480GiB**，还要加保留权重、激活和临时网格。

| 配置 | 本实现状态 |
|---|---|
| 2×B300，约 288GB/卡 | 原实验已完成全 40 层、80 个专家权重张量的两步训练与导出重放；教学入口尚未重新完成整模型运行 |
| 8×RTX 5090，32GB/卡 | 总显存约 256GiB，无法容纳当前完整 FP32 训练状态；启动检查会拒绝，避免盲目 OOM |
| 8×5090 + 大内存 CPU | 需要增加 CPU optimizer/gradient offload 和分片优化器；**本代码没有实现或验证** |
| 单张 5090 | 原实验完整 122B MoE 量化推理可运行；不等于能在单卡上训练全部 122B 权重 |

显存不会自动成为一块连续的大内存。本实现把完整层放到指定卡上，需每卡同时容纳该卡对应的训练状态；它没有 DDP/FSDP/ZeRO，也没有 CPU 缓存或权重卸载。

NVIDIA 规格：[RTX 5090 32GB](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)、[HGX B300 288GB/卡](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)。CPU 内存不是上述 480GiB GPU 状态的替代品。检查点逐张量拷贝到 CPU，35B 单个最大张量及两个动量约需 6GiB 主机临时空间；此外要预留 Python、数据和文件缓存空间。

完整优化器检查点约 **360GiB/份**。默认每 512 步保存一份，4096 步约 2.8TiB 检查点空间，另加原模型、约 66GiB 教师目标和输出。代码保留所有检查点。磁盘较小可增大 `--checkpoint-every`，但中断后丢失的工作会更多。

## 在远程 Linux CUDA 机器上运行

已用的实验环境：Python 3、Torch 2.11.0 + CUDA 13.0、Transformers 5.12.1。`requirements.txt` 固定了 Torch/Transformers 版本；CUDA wheel 请按机器环境选择，已有匹配环境可直接复用。代码不包含 SSH、机器地址、令牌、模型或数据。

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' python test_equations.py
```

先准备本地完整模型目录，例如 `models/Qwen3.5-35B-A3B`，须包含配置、tokenizer 和 safetensors 分片。`texts.jsonl` 每行是 `{"text": "一段足够长的训练文本"}`；请提供足量、合法可用的数据。原实验使用 OpenThoughts 推理与中英文聊天混合数据；这个示例接受你自己的文本，不自动下载该数据集。

```bash
python prepare_data.py \
  --model models/Qwen3.5-35B-A3B --jsonl texts.jsonl \
  --output data/tokens.pt

CUDA_VISIBLE_DEVICES=0,1 python prepare_teacher.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --output data/teacher --devices 0,1

# 先跑两步，验证你的环境、文件路径和显存。
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --teacher data/teacher --output outputs/control --steps 2 --devices 0,1

# 完整的一次遍历；输出目录必须是新目录。
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --teacher data/teacher --output outputs/full --steps 4096 --devices 0,1
```

可以直接使用已有 `tokens.pt`，需包含二维 `train` 和 `val` 整数 token 张量，二者序列长度一致，且 tokenizer 与模型匹配。提供时请自行确认训练与验证来源分离；本工具只对本地文本精确去重并按文档划分，不声称语义去重或评测去污染。

恢复时使用新输出目录，并保持模型、token 数据和教师目录一致：

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --teacher data/teacher --output outputs/resumed --steps 4096 --devices 0,1 \
  --resume outputs/full/checkpoint-000512
```

恢复包含 FP32 主权重、两个动量、Adam 步数/超参数、CPU/CUDA RNG 和固定数据顺序，加载前校验每个状态文件 SHA256。更换 GPU 分片拓扑的恢复未验证，请保持相同设备数。

## 输出与验证范围

`packed/` 保存三值代码、BF16 尺度、矩阵形状和 SHA256 清单。非专家参数继续来自原模型。它是教学用专家权重打包产物，**不是可直接交给 vLLM/llama.cpp 的完整部署模型**；推理适配器和 CUDA Graph 加速尚未收录。

已通过远程 CPU 小规模测试：三值编码往返、非 5 整倍数的 padding、STE 梯度、原生专家完整梯度、分块 KL 的数值/梯度、检查点恢复后下一步参数和动量完全一致、损坏检查点在改权重前拒绝。

原实验另有完整 35B 两步 QAT、全部 80 个张量变化、导出后两组 4K 原生整模型重放一致的证据。**教学整理后的整模型训练、长程稳定性和独立能力评测还未完成**。训练/验证损失改善不能替代代码、数学、中文和格式能力与原模型的独立对比。
