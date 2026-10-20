# Q38：从小实验到全模型恢复训练

[课程首页](README.md) · [只运行公开权重](../../docs/DEPLOYMENT.md#27b) · [实验历史](HISTORY.md)

按下面顺序运行，所有命令从仓库根目录执行。使用Linux和[训练环境](../../docs/REPRODUCING.md#1-环境和资源)；原始模型、数据和输出放在仓库外。每轮使用新的输出目录，GPU命令前确认卡已协调空闲；裸命令不会自动预约GPU或提供私有调度器的限时保护。

<a id="mechanisms"></a>

## 1. 先验证小规模机制

安装根目录依赖后，先做无需27B权重的CPU检查：

```bash
CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/test_core.py
CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/test_native.py
python courses/qwen38_bonsai2/test_source.py
CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/test_recovery_checkpoint.py
```

覆盖旋转、embedding逆旋转、三值打包、STE梯度、小模型导出重载，以及完整状态保存恢复后的下一次更新。这些是小规模机制检查，不代表完整27B质量。

有空闲GPU后检查CPU卸载与磁盘梯度：

```bash
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/test_offload.py
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/test_buffered_staging.py
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/test_mapped_gradients.py
```

<a id="source"></a>

## 2. 核对原始模型和资源

原始模型固定为 `Qwen/Qwen3.8-27B` 的提交 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。下载该版本的权重、配置和tokenizer到独立目录，然后运行：

```bash
python courses/qwen38_bonsai2/verify_source.py \
  --model /path/to/Qwen3.8-27B --output original-source-proof.json

CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/plan.py \
  --config /path/to/Qwen3.8-27B/config.json \
  --index /path/to/Qwen3.8-27B/model.safetensors.index.json \
  --output qwen38-meta-plan.json
```

[source-files.json](source-files.json)固定27个原始文件身份。验证器只读本地文件；形状审计在meta设备构造模型，不加载权重。

| 资源 | 本课程的预算 |
|---|---|
| GPU | 一张协调空闲的32GB RTX 5090，执行前后向 |
| CPU内存 | 映射梯度入口准入约155.04GiB，再加容器已有匿名／共享内存 |
| 32步工作盘 | 原始模型之外约200.19GiB，包含梯度文件、检查点预算、导出与余量 |
| 96步工作盘 | 原始模型之外约237.44GiB，新检查点预算120GB；保留旧检查点 |

这些是准入预算，不是实测峰值或最低配置。FP32主权重放CPU，约100.10GiB梯度逻辑文件放Linux本地磁盘；文件映射仍会占用内核缓存。使用本地文件系统，避免NFS。标准FP32主权重、梯度和两个Adam动量仅训练矩阵就需400.39GiB；这里采用Adafactor，不能据此推断标准Adam训练也能放下。

<a id="train32"></a>

## 3. 用WikiText完成32步训练

准备固定数据：

```bash
python -m pip install -r courses/qwen38_bonsai2/requirements-data.txt
CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/prepare_wikitext.py \
  --model /path/to/Qwen3.8-27B --data-root /path/to/qwen38-data \
  --output /path/to/qwen38-batches.json
```

读取[固定版本WikiText](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)的32段train和8段validation，每段128token；不读取test。文件按官方SHA校验，输出记录段落索引与输入ID身份。下载受限时可用`--endpoint https://hf-mirror.com`，或把同一版本的`train.parquet`、`validation.parquet`放入数据目录并加`--local-only`；校验不变。

确认原始文件和前述测试通过后：

```bash
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/recovery_pilot_mapped.py \
  --model /path/to/Qwen3.8-27B --data /path/to/qwen38-batches.json \
  --output /path/on/local-disk/qwen38-mapped --steps 32 --lr 0.00001
```

训练更新全部402个FP32主权重，前向投影到旋转三值网格，保留参数冻结。优化器为无一阶动量Adafactor，4步预热至`1e-5`；每8步验证，平均KL超过初值1.5倍时停止更新。32段训练只有4096token，属于小型恢复实验。

梯度先完整累积，再更新；更新完成后通过`MADV_REMOVE`回收自有稀疏梯度文件块。导出后尝试独立生成，再保存完整检查点并恢复检查。此入口只在最后保存，没有命令行断点续训选项。

检查点对原始旋转FP32主权重做逐位异或差分后gzip，保存Adafactor与随机状态；恢复必须使用相同原始基底。这是无损存储压缩。32步产物已完成独立恢复，但尚未证明与原进程不中断执行的下一次整模型更新逐位相同。

<a id="verify"></a>

### 独立检查恢复与生成

从自己的运行记录取得下列SHA256；不要用教学记录的哈希替代本地产物身份：

```bash
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/verify_saved_checkpoint.py \
  --model /path/to/Qwen3.8-27B --data /path/to/qwen38-batches.json \
  --checkpoint /path/on/local-disk/qwen38-mapped/checkpoint-final \
  --packed /path/on/local-disk/qwen38-mapped/packed \
  --output /path/to/new-restore-check --expected-step 32 \
  --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA256" \
  --packed-manifest-sha256 "$PACKED_MANIFEST_SHA256" \
  --teacher-cache-sha256 "$TEACHER_CACHE_SHA256"

CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/diagnose_prompts.py \
  --packed /path/on/local-disk/qwen38-mapped/packed \
  --tokenizer /path/to/Qwen3.8-27B \
  --prompts courses/qwen38_bonsai2/diagnostic-prompts.json \
  --output /path/to/new-eight-prompt-check
```

恢复器限于本课程固定模型、g128／1024旋转、Adafactor、lr与最多32步设置。它核验402个FP32主权重、优化器和随机状态，并比较恢复模型与对应packed模型的固定前向。它不会继续训练，也不适用于任意检查点。

可选原始BF16同题对照：

```bash
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/diagnose_original_prompts.py \
  --model /path/to/Qwen3.8-27B \
  --prompts courses/qwen38_bonsai2/diagnostic-prompts.json \
  --quantized-diagnostics /path/to/new-eight-prompt-check/summary.json \
  --quantized-diagnostic-sha256 "$QUANTIZED_DIAGNOSTIC_SHA256" \
  --output /path/to/new-original-reference
```

它检查输入token一致。原始BF16使用CPU权重卸载，只用于质量对照，耗时不能作为纯GPU加速比分母。八条开发题的非空输出、文本检查或损失下降都不能替代语义审阅和独立未见评测。

<a id="continue96"></a>

## 4. 从32步检查点继续到96步

**这是固定旧32步产物的续训实验入口，完整训练配方尚未通过质量验收。** 它要求相同原模型、旧数据／教师身份、恢复证明及402个优化器的step32状态。旧教师缓存SHA固定于源码；换用其他检查点需要另行适配与验证。

[中英文教学上下文](mixed-contexts.json)提供32条train和8条validation，与原WikiText交错后有64条训练序列、16条验证序列。训练共5687token，长度28–128，不填充。数据设计参考过已见开发题，因此这些题不能再用来证明独立泛化。

```bash
CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/prepare_mixed_contexts.py \
  --model /path/to/Qwen3.8-27B --wikitext /path/to/qwen38-batches.json \
  --contexts courses/qwen38_bonsai2/mixed-contexts.json \
  --diagnostic-prompts courses/qwen38_bonsai2/diagnostic-prompts.json \
  --output /path/to/new-mixed-batches.json

CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/prepare_mixed_teacher.py \
  --model /path/to/Qwen3.8-27B --data /path/to/new-mixed-batches.json \
  --data-sha256 "$MIXED_DATA_SHA256" --output /path/to/new-mixed-teacher

CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/test_mixed_objective.py
CUDA_VISIBLE_DEVICES='' python courses/qwen38_bonsai2/test_recovery_checkpoint.py

CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/continue_mixed.py \
  --model /path/to/Qwen3.8-27B --data /path/to/new-mixed-batches.json \
  --data-sha256 "$MIXED_DATA_SHA256" \
  --teacher-cache /path/to/new-mixed-teacher/teacher_states.pt \
  --teacher-cache-sha256 "$MIXED_TEACHER_CACHE_SHA256" \
  --resume-checkpoint /path/on/local-disk/qwen38-mapped/checkpoint-final \
  --resume-manifest-sha256 "$ORIGINAL_CHECKPOINT_MANIFEST_SHA256" \
  --resume-data /path/to/qwen38-batches.json \
  --resume-packed /path/on/local-disk/qwen38-mapped/packed \
  --resume-proof /path/to/new-restore-check/summary.json \
  --resume-proof-sha256 "$ORIGINAL_RESTORE_PROOF_SHA256" \
  --output /path/on/local-disk/new-mixed-run --steps 64
```

从准备结果与自己的运行清单取得各SHA256。续训保留Adafactor状态，以`1e-5`继续，不重新warmup；新更新数与累计步数分别记录。每8步分别检查对话和WikiText组，任一组KL超过各自初值1.5倍时停止更新。训练顺序固定，损失包含所有实际位置，没有答案掩码或硬标签。

已完成的这轮新增64步、累计96步；81.34GB检查点实际回读和对应packed重载前向一致。随后独立生成首题触发显存保护，0/8题完成，新版本质量待验证。最终对话KL下降，WikiText KL高于初值；完整轨迹与失败见[历史记录](HISTORY.md#96步保存与恢复通过生成因显存保护中止)。

<a id="method"></a>

## 方法与代码地图

| 项目 | 本课程实现 |
|---|---|
| 模型范围 | 64层稠密文本骨干，402个矩阵；不训练视觉塔和MTP |
| 量化 | g128三值、FP16尺度、1024宽度固定符号／Hadamard旋转 |
| 保留路径 | `in_proj_a/b`、conv1d、A_log/dt_bias、归一化等；本实验选择冻结 |
| 训练 | FP32主权重、STE，`KL + 0.1 × 状态MSE`，CPU Adafactor |
| 导出／推理 | 自定义base-3打包；参考实现逐矩阵展开，尚无融合低比特内核 |

对行向量令`R = H·S/√1024`，同时将权重变为`W·Rᵀ`、输入变为`x·Rᵀ`。未量化时两侧旋转相消；量化后产生训练要恢复的误差。embedding查得旋转基底向量后还需逆旋转，符号与Hadamard的顺序不能交换。

表示与保留边界参考[官方白皮书](https://github.com/PrismML-Eng/Bonsai-demo/blob/330450a1a120bbfb2dd3b0f0bbc4315a7dbc8120/bonsai-2-27b-whitepaper.pdf)；固定符号、Lloyd初始化、损失与优化器是本项目选择，不是官方完整配方。该产物不能直接交给官方GGUF运行时。

阅读顺序：`rotation.py` → `ternary.py` → `policy.py` → `offload.py` → `mapped_gradients.py` → `recovery_pilot_mapped.py` → `recovery_checkpoint.py` → `continue_mixed.py`。确切源码身份见[source-provenance.json](source-provenance.json)，早期两步SGD、真实块对照与失败记录见[HISTORY](HISTORY.md)。
