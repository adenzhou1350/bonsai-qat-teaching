# 从下载到实际使用：单张5090部署

先选模型，再安装对应环境、下载完整产物、做CPU文件校验，最后启动推理。下面从仓库根目录执行，路径可自行替换。所有模型都是文本实验版，不能直接交给标准`AutoModel`、llama.cpp或未适配的`vllm serve`。

## 模型区别和选择

| 模型 / 路线 | 训练了什么 | 推理文件体积 | 单卡实测路径 | 适合先试什么 |
|---|---|---:|---|---|
| Qwen3.5-35B-A3B，R35-8192 | 冻结INT2专家基座，训练rank16补偿；4096+4096步 | 约15.86GB | CUDA Graph单请求；4K固定解码约49.85token/s | 在这些实验产物里，优先尝试终端问答；新便携入口GPU复验待完成 |
| Qwen3.5-122B-A10B，native4095 | 三值专家、dense/embedding4bit，冻结基座训练rank8；4096步 | 约30.20GB | 自定义Graph；4K固定解码约25.84token/s | 研究大MoE压缩，显存余量较小 |
| Qwen3.8-27B，旧step32 | 402个稠密矩阵的旋转三值全矩阵QAT | 约5.85GB | 每层展开权重的参考实现，约0.69token/s | 教学与数值验证；当前没有高吞吐融合实现 |
| Qwen3.8-27B，新step96 | 在旧32步检查点上追加64步混合上下文训练 | 约5.85GB | 保存与恢复通过，独立生成未完成 | 尚未发布，等待空闲卡补测 |
| Qwen3.5-35B-A3B，T35 | 更新全专家FP32主权重的QAT | 单份训练检查点约386.55GB | 最新完整保存2560步，长程已暂停 | 训练教学；不是上面8192步的可部署文件 |

35B/122B是MoE，总参数与每token激活参数不同；不能用“122B比35B大”直接推导速度或质量。35B的8192步补偿不是全专家QAT，也不是官方Bonsai格式。四组开发评测没有全部达到原模型，发布不等于质量无损。

体积用十进制GB；显存表另用GiB。速度来自历史固定64步Graph测速，排除预填充/编译/录制，即使EOS也继续，不是交互端到端速度、并发吞吐或当前新入口的重新测量。详见[35B记录](experiments/35B.md)、[历史推理](experiments/INFERENCE.md)及[结果口径](RESULTS.md)。

## 先安装环境

GPU推理使用Linux、Python3.12、单张空闲32GB RTX5090。建议至少32GB主机内存，64GB更宽裕；这属于部署余量建议，不是测得的最小配置。下载35B/122B分别预留至少25GB/40GB磁盘，环境和编译缓存另外留空间。**只跑推理不需要B300、不需要原始BF16大权重或完整优化器检查点。**

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja

# 安装记录中的CUDA13.0 wheel，避免默认选到另一版Torch。
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt

# 35B/122B的历史解码使用原生卷积更新；在Torch安装后编译。
# 需要可用的CUDA工具链和兼容编译器。27B参考路径不要求这一扩展。
python -m pip install causal-conv1d==1.7.0 --no-build-isolation
nvidia-smi
python scripts/check_deployment_environment.py --model 35b
```

记录中的组合是Torch2.11.0+cu130、Transformers5.12.1、Triton3.6.0、causal-conv1d1.7.0。依赖文件固定版本，但不是完整容器镜像或已在全新机器逐项验证的环境锁。检查器报告包版本和NVML状态，不加载权重、不初始化CUDA；安装成功也不能替代后面的实际生成。CUDA驱动与系统编译工具不由本仓库自动安装。参考[PyTorch官方安装表](https://pytorch.org/get-started/previous-versions/)和[卷积扩展源码](https://github.com/Dao-AILab/causal-conv1d)。

## 下载身份与发布状态

| 产物 | Hugging Face仓库 | 固定权重标签 | 当前状态 |
|---|---|---|---|
| R35-8192 | `aden1350/Qwen3.5-35B-A3B-INT2-Rank16-8192-Experimental` | `recovery-step8192` | 发布中，尚未公开验证 |
| 122B | `aden1350/Qwen3.5-122B-A10B-Ternary-Rank8-Experimental` | `native4095-step4096` | 发布中，尚未公开验证 |
| 27B旧32步 | [模型页](https://huggingface.co/aden1350/Qwen3.8-27B-BonsaiStyle-Ternary-QAT-Experimental) | `qat-step32` | 已公开，固定提交`12585f8e5d50c847071d8564c6f6aa7f4950a4ef` |

35B/122B下面的下载命令在对应标签完成公开后使用；私有阶段出现404不代表安装错误。权重标签固定历史文件，更新后的教学入口及运行说明来自本仓库。下载时保留模型仓库的全部目录，尤其`release-files.json`、`release-provenance.json`、`packed/`和tokenizer。不要只下载若干`.pt`，也不要把模型指向原BF16目录。

## 35B：一次提问，然后常驻终端聊天

```bash
hf download aden1350/Qwen3.5-35B-A3B-INT2-Rank16-8192-Experimental \
  --revision recovery-step8192 --local-dir models/r35-8192

# 纯CPU校验全部81个张量文件与配置；不导入Torch。
python experimental35/infer.py --model models/r35-8192/packed --verify-only

CUDA_VISIBLE_DEVICES=0 python experimental35/infer.py \
  --model models/r35-8192/packed --prompt '请用三点解释模型量化的作用。' \
  --context 4096 --max-new-tokens 256 --output outputs/r35-first.json

# 同一进程保留模型与Graph，逐次串行回答；/reset清空对话，/exit退出。
CUDA_VISIBLE_DEVICES=0 python experimental35/infer.py \
  --model models/r35-8192/packed --chat --context 4096 --max-new-tokens 256
```

默认greedy、不启用thinking，生成遇EOS停止。`--context`约束聊天模板后的输入加输出预算；多轮超过预算会提示`/reset`，不偷偷截断历史。可以用`--context 8192`或`16384`扩大缓存；最长受限的是已测数值路径，不代表16K能力已验收。输出可增到`--max-new-tokens 1024`，但历史性能表固定64步，不能据此保证长回答速度或质量。

新`experimental35`入口提供公开文件身份检查，替代历史私有目录的实验准入；专家内核、Graph类及模型组装数值语句与历史版本核对一致。**新便携入口目前CPU/源码核验，完整GPU复验待空闲卡**；历史8192步产物本身已完成独立重载、5090缓存及Graph验证。这两个状态分别记录在[来源清单](../experimental35/provenance.json)，不把旧GPU结果冒充新入口已跑过。

## 35B怎么配置，才能用好现有实现

| 使用目标 | 设置 | 依据和代价 |
|---|---|---|
| 默认交互 | 一张卡、一个进程、`--chat --context 4096` | 保留模型和Graph，避免每次冷启动；每轮仍需预填充，首轮还要编译/录制 |
| 更短响应 | 缩短实际输入、减少输出上限 | 128/512-token历史固定解码约59.33/58.12token/s；不能只设置小缓存却输入更多token |
| 更长输入 | `--context 8192`或`16384` | 历史固定解码约42.25/32.71token/s，显存与首token等待增加 |
| 多用户 | 先按串行请求使用 | Graph共享工作区不能直接并发；未发布该8192产物的vLLM服务适配 |

不要开多个进程试图“吃满”显存；模型会重复驻留。不要关闭文件校验、把量化文件改名伪装成标准权重，或开启未经本产物验证的FP8、FlashAttention、fused GDN等选项来套用旧数值结论。这里采用现有已验证内核路径，**没有证明这是5090的理论最大吞吐**。

以下在自己的空闲卡测Graph，输出每档64步的Graph/静态token一致性、解码速度和PyTorch峰值。它会越过EOS，只用于工程计时：

```bash
CUDA_VISIBLE_DEVICES=0 python experimental35/infer.py \
  --model models/r35-8192/packed --benchmark --contexts 128 512 4096 8192 16384 \
  --output outputs/r35-graph-benchmark.json
```

历史R35在128/4K/16K的峰值allocated约14.85/15.61/18.26GiB，包含缓存初始化与Graph录制，不是整卡NVML峰值；另外保留桌面、驱动、其他工作区和编译余量。完整数字见[五档记录](experiments/2026-10-09-2156.json)。

## 122B：公开清单用专用入口

```bash
hf download aden1350/Qwen3.5-122B-A10B-Ternary-Rank8-Experimental \
  --revision native4095-step4096 --local-dir models/r122
python experimental122/infer_hf.py --model models/r122/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python experimental122/infer_hf.py \
  --model models/r122/packed --prompt '请用中文解释模型量化。' \
  --max-new-tokens 128 --output outputs/r122-first.json
```

`infer_hf.py`适配脱敏后的公开清单，471个张量文件与历史相同；旧`infer.py`以及旧HTTP入口固定另一份清单，不能直接替换目录。当前公开权重推荐上述单请求路径，**不提供一个已经验证能加载公开清单的HTTP服务命令**。历史HTTP四路的23.30输出token/s是合计吞吐且有输出差异，不是这个CLI的一人速度。

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/infer_hf.py \
  --model models/r122/packed --benchmark --contexts 128 512 4096 \
  --output outputs/r122-graph-benchmark.json
```

先测短输入再扩大上下文。历史native4095固定Graph解码128/4K/16K约28.75/25.84/19.01token/s；16K峰值allocated约30.26GiB，32GB卡余量很小。不要直接让多个权重进程或桌面任务共用；缩短上下文可减少缓存，但不会减少约28.10GiB常驻文本权重。

## 27B：自定义问题，而不是只能跑固定提示

```bash
hf download aden1350/Qwen3.8-27B-BonsaiStyle-Ternary-QAT-Experimental \
  --revision qat-step32 --local-dir models/q38-step32
python courses/qwen38_bonsai2/infer_prompt.py \
  --packed models/q38-step32/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/infer_prompt.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer \
  --prompt '请用一句话介绍北京。' --max-new-tokens 32 --output outputs/q38-first.json
```

新入口支持任意短文本，含模板的输入最多128token，输出最多64token；使用原参考加载器和相同greedy/KV循环。原固定题入口`infer_control.py`保留用于对照。新入口的CPU文件校验可以现在执行，GPU生成尚待空闲卡复验。27B体积最小，但展开权重造成运行峰值远高于5.45GiB常驻存储，历史约24.49GiB allocated；不能按文件大小估算同时能开几个进程。目前没有本项目27B的高速融合或HTTP服务实现。

## 如何判断已经可用

1. 下载固定标签，`--verify-only`实际通过；不要只看CLI退出或文件数量。
2. 实际生成能结束，阅读JSON里的文本、token、EOS及资源记录。回答非空不等于语义正确。
3. 多次使用自己的中英文、格式和长短输入，检查错误与延迟；再决定是否适合用途。本文的开发题不代表独立能力验收。
4. 分别测冷启动、首token和热解码。测速中跳过EOS的固定64步不能当作日常聊天的端到端吞吐。

## 常见问题

| 现象 | 处理 |
|---|---|
| `Failed to initialize NVML: Driver/library version mismatch` | 先协调修复宿主机驱动；本工具不会重启、卸载驱动或停止桌面会话 |
| `causal_conv1d`缺失、ABI/undefined symbol | 在已装好对应Torch/CUDA工具链的隔离环境重建1.7.0扩展，再检查；不把静默回退当成历史同路径 |
| Transformers/Triton校验或API错误 | 回到记录版本；不同版本是否可用需另测，不删除断言绕过 |
| OOM或Graph录制失败 | 检查其他进程及整卡空闲，先用短上下文、单请求；不要只看packed文件体积 |
| `Expected ... release`或SHA不符 | 检查模型/标签/路径；35B、122B、27B入口不能互换，脱敏清单与旧清单有意不同 |
| `--output`已存在 | 用新路径保留上次证据 |
| 首次很慢 | 下载、权重加载、Triton编译、预填充和Graph录制都不在历史热解码数字里；优先用常驻35B聊天模式 |

所有权重只放模型Hub，GitHub保留代码和记录。完整训练检查点、教师缓存不需要下载；暂停的35B/122B训练不会因为发布或使用权重而恢复。
