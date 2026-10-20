# 单卡 5090：下载与运行

按顺序完成 **安装环境 → 选择模型 → 复制命令**。只需推理权重，无需 B300 或训练检查点。

## 1. 安装一次环境

需要 Linux、Python 3.12、空闲的 32GB RTX 5090。主机内存建议32GB以上；35B/122B下载分别预留25GB/40GB磁盘，环境另计。

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt
```

35B/122B还需安装卷积扩展；27B跳过此步。编译需要兼容的 CUDA 工具链：

```bash
python -m pip install causal-conv1d==1.7.0 --no-build-isolation
nvidia-smi
python scripts/check_deployment_environment.py --model 35b
```

检查122B时把 `35b` 改为 `122b`。以上是记录中的依赖组合；新机器环境和便携入口仍需实际生成验证。

## 2. 选择模型

| 产物 | 适合用途 | 下载体积 | 当前状态 |
|---|---|---:|---|
| R35-8192 | 终端聊天；INT2 + rank16 补偿 | 15.86 GB | 发布中，尚未公开验证 |
| 122B | 大 MoE 压缩实验；三值 + 4bit + rank8 | 30.20 GB | 发布中，尚未公开验证 |
| 27B旧32步 | 旋转三值 QAT 教学；参考推理较慢 | 5.85 GB | 已公开 |

**35B/122B完成公开后才能下载**，当前私有阶段返回404属于预期。固定提交与发布状态见[发布目录](model-releases.json)。

全部为自定义文本量化格式，不能直接用标准 `AutoModel`、llama.cpp 或 `vllm serve`。量化质量尚未全面达到原模型；27B的96步候选不在本次发布中。

## 3. 展开对应命令

<a id="35b"></a>
<details open>
<summary><strong>35B：下载后常驻聊天</strong></summary>

```bash
hf download aden1350/Qwen3.5-35B-A3B-INT2-Rank16-8192-Experimental \
  --revision recovery-step8192 --local-dir models/r35-8192
python experimental35/infer.py --model models/r35-8192/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python experimental35/infer.py \
  --model models/r35-8192/packed --chat --context 4096 --max-new-tokens 256
```

输入自己的问题即可。`/reset`清空对话，`/exit`退出；默认greedy、遇EOS停句。

只问一次，把 `--chat` 换成 `--prompt '请用三点解释模型量化。'`，可加 `--output outputs/answer.json` 保存结果。

默认4K适合先试；需要更长输入时改为 `--context 8192` 或 `16384`。一个进程串行使用，不同时加载多份模型。历史权重已在5090验证，**新便携入口的GPU复验待完成**。

</details>

<a id="122b"></a>
<details>
<summary><strong>122B：自定义问题</strong></summary>

```bash
hf download aden1350/Qwen3.5-122B-A10B-Ternary-Rank8-Experimental \
  --revision native4095-step4096 --local-dir models/r122
python experimental122/infer_hf.py --model models/r122/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python experimental122/infer_hf.py \
  --model models/r122/packed --prompt '请用中文解释模型量化。' \
  --max-new-tokens 128 --output outputs/r122-first.json
```

公开清单用 **`infer_hf.py`**。旧 `infer.py` / HTTP入口要求另一份清单，不能直接加载公开包；本页不提供已验证的公开包HTTP服务。

历史常驻权重约28.10GiB，32GB卡余量较小。先测短问题，显卡保持独占；输出选择新文件名。

</details>

<a id="27b"></a>
<details>
<summary><strong>27B：短提示教学验证</strong></summary>

```bash
hf download aden1350/Qwen3.8-27B-BonsaiStyle-Ternary-QAT-Experimental \
  --revision qat-step32 --local-dir models/q38-step32
python courses/qwen38_bonsai2/infer_prompt.py --packed models/q38-step32/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/infer_prompt.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer \
  --prompt '请用一句话介绍北京。' --max-new-tokens 32 --output outputs/q38-first.json
```

含模板的输入最多128 token，输出最多64 token。参考路径逐层展开权重，历史约0.69 token/s、峰值约24.49GiB；5.85GB文件体积不等于运行显存。新自定义提问入口已通过CPU文件校验，GPU复验待完成。

</details>

## 调好配置再使用

35B先用常驻聊天、4K上下文和256输出上限，避免每次重新加载。更长上下文会增加显存和首token等待；目前没有证明理论最大吞吐或长上下文质量。

需要8K/16K、完整显存和速度表、测速命令或排障时，查看[部署参数参考](DEPLOYMENT_REFERENCE.md)。运行后检查实际回答和停句是否满足你的用途；源码、下载校验和速度记录不代替质量验证。
