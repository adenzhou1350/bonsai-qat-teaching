---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
- zh
tags:
- quantized
- ternary
- custom-code
- experimental
- bonsai-style
---
# Qwen3.8-27B · 三值 QAT · 32步

27B稠密文本模型的旋转三值QAT，402个矩阵，32次更新。单张RTX 5090短提示热解码已从约0.69提升到 **9.37 token/s**，四轮八题输出与逐步logits一致。教学实验，质量仍未通过，本轮优化已停止。

| 下载体积 | 短提示测量 |
|---|---|
| 约5.85GB | 9.37 token/s；首token计算中位数0.305秒；峰值allocated 8.12GiB |

计时为batch=1、greedy、动态KV缓存、每题最多32个新token；不含加载、tokenization、logits回传/哈希或输出，非端到端吞吐。显存是PyTorch allocated，非整卡占用。[加速原理与完整计时](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/ed7267283fe5a0e70a5c997d534bc2113f31ec7a/courses/qwen38_bonsai2/inference/README.md)。

后续同权重私有GDN Graph实验测得约13.02 token/s，便携Graph入口尚未GPU验证，未发布；以下部署命令仍使用9.37路径。[优化收尾记录](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/main/docs/experiments/2026-10-20-q38-inference-closeout.md)。

## 开始使用

需要Linux、Python3.12和一张空闲32GB RTX5090。[完整部署](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/ed7267283fe5a0e70a5c997d534bc2113f31ec7a/docs/DEPLOYMENT.md#27b)。

<details>
<summary>首次使用：安装环境</summary>

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
git checkout ed7267283fe5a0e70a5c997d534bc2113f31ec7a
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt
nvidia-smi
```

27B不要求卷积扩展；新机器仍需检查驱动、空闲显存与实际输出。

</details>

从教学仓库根目录运行：

```bash
hf download aden1350/Qwen3.8-27B-BonsaiStyle-Ternary-QAT-Experimental \
  --revision qat-step32 --local-dir models/q38-step32
python courses/qwen38_bonsai2/inference/infer.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer --verify-only
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/inference/infer.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer \
  --mode decode_rotate --prompt '请用一句话介绍北京。' \
  --max-new-tokens 32 --output outputs/q38-first.json
```

输入含模板最多128 token，输出最多64 token；输出文件用新名字。`--mode reference`可回退。共用加载/生成代码已由便携benchmark覆盖，单独交互CLI未直接GPU调用。

## 使用边界

量化定义、格式遵循仍有退化，八条开发题不代表完整能力评测。96步候选不包含在本发布中；[32步输出与测速证据](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/ed7267283fe5a0e70a5c997d534bc2113f31ec7a/docs/experiments/2026-10-20-Q38-step32-fast-portable.json)。

独立Bonsai风格文本实验，使用自定义格式，不能直接加载到标准Transformers、vLLM或llama.cpp。固定权重标签`qat-step32`不变；[发布身份](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/ed7267283fe5a0e70a5c997d534bc2113f31ec7a/docs/model-releases.json)。

## 来源与许可

基础模型：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，Apache-2.0，原许可随包附带。运行时代码遵循随附许可证。此包仅含推理产物；完整训练检查点和教师缓存不分发。
