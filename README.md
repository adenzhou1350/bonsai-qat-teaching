# Bonsai QAT Teaching

[![公开文件检查](https://github.com/adenzhou1350/bonsai-qat-teaching/actions/workflows/public-evidence.yml/badge.svg?branch=main)](https://github.com/adenzhou1350/bonsai-qat-teaching/actions/workflows/public-evidence.yml)

学习低比特量化，也动手把大模型放进一张 5090。这里提供教学代码、实验权重的运行入口，以及可核对的结果。

**[部署模型](docs/DEPLOYMENT.md) · [从第一课开始](docs/LABS.md) · [复刻训练](docs/REPRODUCING.md)**

## 选一个模型

| 模型 | 方法 | 下载体积 | 单卡5090速度记录 |
|---|---|---:|---|
| [35B · 8192步](experimental35/README.md) | INT2 专家 + rank16 补偿 | 15.86 GB | 历史热解码：短上下文59.33、4K约49.85 token/s |
| [122B · 4096步](experimental122/README.md) | 三值专家 + 4bit 稠密层 + rank8 补偿 | 30.20 GB | 历史短请求CLI约21.72–24.00 token/s |
| [27B · 32步](courses/qwen38_bonsai2/README.md) | 旋转三值 QAT | 5.85 GB | 公开路径短提示热解码约9.37 token/s |

三款权重均已公开。上表测试口径不同，不能直接计算模型间加速比；35B新便携入口GPU复验未完成。35B/122B 是 MoE，每token约启用3B/10B参数；27B是稠密模型。按[部署指南](docs/DEPLOYMENT.md)选择加载器，不能直接使用标准 vLLM 或 llama.cpp。

这是独立的 Bonsai 风格教学实现，质量尚未全面达到原模型。27B额外Graph实验测得 **13.02 token/s**，便携入口尚未GPU验证；当前部署仍用9.37路径。本轮优化已停止，详见[收尾记录](docs/experiments/2026-10-20-q38-inference-closeout.md) · [测速口径](docs/DEPLOYMENT_REFERENCE.md) · [固定下载版本](docs/model-releases.json)。

<a id="cpu-start"></a>

## 不下载模型，也能开始

先玩[浏览器三值打包实验](https://adenzhou1350.github.io/learn/bonsai-low-bit/#ternary-demo)，或在 Linux CPU 上检查量化、梯度和保存恢复：

```bash
git clone https://github.com/adenzhou1350/bonsai-qat-teaching.git
cd bonsai-qat-teaching
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES='' .venv/bin/python test_equations.py
```

成功会输出 `PASS: ternary grid, identity STE, packing, ...`。接着按[动手课程](docs/LABS.md)读代码、改参数。

## 按需深入

- **学原理**：[量化与训练方法](docs/METHOD.md) · [27B QAT 课程](courses/qwen38_bonsai2/README.md)
- **复刻实验**：[训练与资源要求](docs/REPRODUCING.md) · [122B 数据课程](courses/122b/README.md)
- **查结果**：[质量与速度](docs/RESULTS.md) · [实验档案](docs/experiments/README.md) · [公开证据校验](docs/PUBLIC_EVIDENCE.md)

问题反馈请附环境、命令和实际输出：[Issues](https://github.com/adenzhou1350/bonsai-qat-teaching/issues)。

原创代码与文档采用 [MIT](LICENSE)；模型、数据和第三方代码遵循各自许可，见[来源说明](THIRD_PARTY_NOTICES.md)。权重从 Hugging Face 单独下载。

<details>
<summary>旧章节链接</summary>

<a id="从-cpu-上的三值实验走到大模型训练与单卡压缩推理"></a>
<a id="可以从这里学到什么"></a>
<a id="再选一条进阶路线"></a>
<a id="有指定权重启动-http-服务"></a>
<a id="当前结果122b-moe单张-32gb-rtx-5090"></a>
<a id="实验状态与质量"></a>
<a id="代码阅读地图"></a>
<a id="一起完善这门课"></a>

旧训练与推理记录见[实验档案](docs/experiments/README.md)，当前使用入口见[部署指南](docs/DEPLOYMENT.md)。

</details>
