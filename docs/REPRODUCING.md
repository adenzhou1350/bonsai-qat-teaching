# 复刻步骤与边界

本页的命令从仓库根目录执行。先确定要复刻的是方程、训练流程、固定权重推理，还是原实验的能力数字；当前支持范围不同。

| 层级 | 当前可做 | 还需要什么 |
|---|---|---|
| CPU 方程与检查点 | 直接运行仓库测试 | 安装匹配依赖 |
| 35B 两步/完整训练流程 | 从原始模型与本地文本运行 | 2×B300；长程已暂停，尚无完整质量验收 |
| 固定 122B 推理与工程测速 | 用仓库入口运行 | 另行提供指定 manifest 对应的 packed 权重 |
| 完全复现已记录的质量数字 | 目前不能仅凭仓库完成 | 精确数据修订、采样/预处理清单、评测题集与评分脚本尚未完整收入 |
| 从原始 122B 训练出同一产物 | 全链路未完成，研究已暂停 | 全部校准、教师、训练与独立重载流程验证后发布 |

## 35B全专家权重QAT

### 1. 环境和资源

采用已验证的 Linux、2×B300 路径。原实验 Torch `2.11.0+cu130`、Transformers `5.12.1`；根目录 [requirements.txt](../requirements.txt) 固定 Torch/Transformers，其他依赖并未形成完整环境锁。CUDA wheel 按机器环境安装，不要把默认 pip wheel 自动等同于原实验环境。

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' python test_equations.py
python -m pip freeze > environment.txt
```

CPU 测试成功应输出 `PASS: ternary grid, identity STE, packing, ...`，不需要加载模型。完整模型两步控制曾核验全部 80 个导出文件及完整优化器检查点、原始 693 个文本参数和保留的 613 个参数；这些完整模型核验是独立实验，不是 CPU 测试的内容。

资源预留：专家训练状态约 480GiB，实测每卡约 247.5GiB；单个完整优化器检查点约 360GiB。默认每 512 步保存，4096 步约 2.8TiB 检查点，另有模型、约 66GiB 教师目标及输出。代码保留所有检查点，减小频率可降低磁盘需求，但会增加中断损失。逐张量保存时还需约 6GiB 最大主机临时空间，另留 Python 与文件缓存余量。

### 2. 模型和数据

模型目录示例 `models/Qwen3.5-35B-A3B`，须包含原始配置、tokenizer 与 safetensors 分片及索引。固定模型修订，记录所有输入文件 SHA256；仅索引相同不能证明每个权重分片相同。

`texts.jsonl` 每行 `{"text":"足够长的训练文本"}`。提供足量、合法可用的数据。

```bash
python prepare_data.py \
  --model models/Qwen3.5-35B-A3B --jsonl texts.jsonl \
  --output data/tokens.pt
```

默认 4096 条 TRAIN、128 条 VAL，每条 4096 token。代码精确去重文本，seed 9117 打乱后将约 1/32 文档留作 VAL，再各自拼接切块。它没有语义去重、来源组去重或评测去污染；数据不足时会报错。教学预处理器与原实验数据采样器不同。

原 35B 实验用了 OpenThoughts 2048 条及中英文 WildChat 各 1024 条 TRAIN；VAL 为 64+32+32 条，按来源组分离。原 122B 恢复训练用了 OpenThoughts 4096 TRAIN/128 VAL。当前没有发布完整数据版本与采样清单，换成自己的 JSONL 可以复刻训练机制，不能声称复制原分数。

可直接使用已有 `tokens.pt`，须含二维整数 `train`/`val` 张量、序列长度一致、tokenizer 匹配，并自行确认来源分离。

### 3. 准备位置匹配的 BF16 教师

```bash
CUDA_VISIBLE_DEVICES=0,1 python prepare_teacher.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --output data/teacher --devices 0,1
```

应得到 `train.bin`、`val.bin`、`metadata.json`。每条缓存形状为 `[4095, hidden_size]`；元数据保存 shape、token 文件、模型索引及缓存文件 SHA。先查元数据再训练。不要混用输入 4096 位置或不同专家后端的旧教师缓存。当前写法按行写盘，已与原写法逐字节对照。

### 4. 两步控制，然后完整一遍

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --teacher data/teacher --output outputs/control --steps 2 --devices 0,1

CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --teacher data/teacher --output outputs/full --steps 4096 --devices 0,1
```

`--steps 2` 只缩短优化器更新数，默认仍验证全部 VAL，教师仍需准备好。输出目录必须不存在；异常中断后保留旧目录供检查，下一次用新目录。示例默认 `--lr 1e-4 --checkpoint-every 512`，完整一遍为 16,773,120 个监督位置。

### 5. 检查产物和恢复

```text
outputs/full/
  validation-before.json
  train.jsonl
  checkpoint-000512/ ... checkpoint-004096/
  validation-after.json
  packed/manifest.json
  packed/<80 个专家张量>.pt
```

训练日志最后应到计划步数，验证条数与 VAL 相同，manifest 应有 80 个矩阵及逐文件 SHA。导出包含三值代码、BF16 尺度与形状；非专家参数继续从原模型读取。它不是完整 Transformers/vLLM 部署检查点，也不能交给 `experimental122/infer.py`。

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model models/Qwen3.5-35B-A3B --tokens data/tokens.pt \
  --teacher data/teacher --output outputs/resumed --steps 4096 --devices 0,1 \
  --resume outputs/full/checkpoint-000512
```

恢复包括 FP32 主权重、两个 Adam 动量、步数与超参数、CPU/CUDA RNG 和固定数据顺序；加载前逐文件校验 SHA。保持同一模型、tokens、教师及设备数，改变 GPU 分片拓扑的恢复未验证。

### 6. 验证结论

两步通过后才启动长程。长程完成后仍需要独立加载、逐层/整模型缓存对照、中文/格式/知识/数学/代码公开开发评测，达到预先规定的标准后才执行保留测试。仓库尚未收入这套独立评分流水线，不能把根目录导出成功叫作质量达标。

## 122B固定权重推理

按 [experimental122/README.md](../experimental122/README.md) 的环境、manifest SHA 和命令执行。明确选择 `old` 或 `native4095`；没有对应 packed 权重时，此路线不可执行。不要换成原始 BF16 权重或不完整产物。

测速固定 64 步，报告解码耗时，排除预填充与 Graph 录制；自然生成按 EOS 停止。比较速度时保持权重、提示、上下文、输出长度和计时口径一致。已发布入口不包含通用 `portable` 或 `--verify-only` 参数，后续候选入口的 CPU 检查不等于已发布 GPU 功能。

## 保存一份自己的复刻记录

记录仓库 commit、环境包版本/CUDA 后端、模型及 tokenizer 修订/文件 SHA、原始数据版本与预处理参数、tokens/教师 SHA、TRAIN/VAL 来源划分、训练超参数、导出 manifest SHA、测试命令与退出码。公开结果还需记录题集、生成上限/EOS、评分规则与不支持样本分母。

[reproducibility-manifest.json](reproducibility-manifest.json) 固定本次教学代码及推理入口身份；文本统一 CRLF→LF 后哈希，以兼容 Windows/Linux 检出。可在仓库根目录核对：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('docs/reproducibility-manifest.json').read_text())
for name, expected in manifest['file_sha256'].items():
    content = Path(name).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(content).hexdigest() == expected, name
print('PASS: teaching source identities')
PY
```

该清单不包含权重或数据分发许可，也不表示所有依赖已经锁定。新增可运行课程须通过原始输入 → 训练 → 导出 → 独立重载的完整控制，再标记“可复刻”。
