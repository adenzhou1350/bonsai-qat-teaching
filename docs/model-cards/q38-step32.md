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

27B稠密文本模型的旋转三值QAT，402个矩阵，32次更新。**当前用于教学验证，不适合日常聊天：参考实现约0.69 token/s。**

| 下载体积 | 单卡参考 |
|---|---|
| 约5.85GB | 历史参考路径峰值约24.49GiB；约0.69 token/s |

权重全在GPU且已启用KV缓存，但每个token仍会重新解包完整矩阵，再做普通矩阵乘。压缩存储已实现，高速融合推理尚未实现；增加训练步数不会自动加速。

## 开始使用

需要Linux、Python3.12和一张空闲32GB RTX5090。[完整部署与模型选择](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/main/docs/DEPLOYMENT.md#27b)。

<details>
<summary>首次使用：安装环境</summary>

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
git checkout 0664e61bcf41f46d88a5d3fe5f6e07af125853e0
python3.12 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt
nvidia-smi
```

27B参考路径不要求卷积扩展；这里固定记录版本，仍需在目标机器完成实际生成检查。

</details>

从教学仓库根目录运行：

```bash
hf download aden1350/Qwen3.8-27B-BonsaiStyle-Ternary-QAT-Experimental \
  --revision qat-step32 --local-dir models/q38-step32
python courses/qwen38_bonsai2/infer_prompt.py --packed models/q38-step32/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/infer_prompt.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer \
  --prompt '请用一句话介绍北京。' --max-new-tokens 32 --output outputs/q38-first.json
```

输入含模板最多128 token，输出最多64 token；输出文件请用新名字。新自定义提问入口已通过CPU文件校验，GPU复验待完成。

## 使用边界

32步权重的独立恢复与八条短提示已验证，但量化定义、格式遵循仍有退化；质量恢复未通过。96步候选不包含在本发布中。

本项目是独立Bonsai风格文本实验，使用自定义格式，不能直接加载到标准Transformers、vLLM或llama.cpp。性能数字来自历史测量，非新入口端到端吞吐；[显存、上下文与测速说明](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/main/docs/DEPLOYMENT_REFERENCE.md)。

[完整实验记录](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/main/docs/experiments/2026-10-20-Q38-checkpoint-verified-eight-prompts.json) · [发布身份](https://github.com/adenzhou1350/bonsai-qat-teaching/blob/main/docs/model-releases.json)

## 来源与许可

基础模型：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，Apache-2.0，原许可随包附带。运行时代码遵循随附许可证。此包仅含推理产物；完整训练检查点和教师缓存不分发。
