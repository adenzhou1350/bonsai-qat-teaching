# Bonsai QAT Teaching

[![公开文件检查](https://github.com/adenzhou1350/bonsai-qat-teaching/actions/workflows/public-evidence.yml/badge.svg?branch=main)](https://github.com/adenzhou1350/bonsai-qat-teaching/actions/workflows/public-evidence.yml)

学习低比特量化，也动手把大模型放进一张 5090。这里提供教学代码、实验权重的运行入口，以及可核对的结果。

**[部署模型](docs/DEPLOYMENT.md) · [从第一课开始](docs/LABS.md) · [复刻训练](docs/REPRODUCING.md)**

## 选一个模型

| 模型 | 方法 | 下载体积 | 当前状态 |
|---|---|---:|---|
| [35B · 8192步](experimental35/README.md) | INT2 专家 + rank16 补偿 | 15.86 GB | 发布中；提供终端聊天入口，便携入口 GPU 复验待完成 |
| [122B · 4096步](experimental122/README.md) | 三值专家 + 4bit 稠密层 + rank8 补偿 | 30.20 GB | 上传中；使用专用单请求入口 |
| [27B · 32步](courses/qwen38_bonsai2/README.md) | 旋转三值 QAT | 5.85 GB | [权重已公开](https://huggingface.co/aden1350/Qwen3.8-27B-BonsaiStyle-Ternary-QAT-Experimental)；适合教学，参考推理较慢 |

35B/122B 是 MoE。三条路线使用不同的自定义格式，须按[部署指南](docs/DEPLOYMENT.md)选择加载器；不能直接使用标准 vLLM 或 llama.cpp。

这是独立的 Bonsai 风格教学实现。量化质量尚未全面达到原模型；27B 的96步候选已保存并通过恢复检查，生成验证仍待完成。方法区别与测量条件见[结果说明](docs/RESULTS.md)，固定下载版本见[发布目录](docs/model-releases.json)。

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
