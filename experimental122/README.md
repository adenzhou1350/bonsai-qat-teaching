<a id="单张-rtx-5090-跑-122b已验证的实验版"></a>

# 122B 实验推理：运行指南

这里提供 Qwen3.5-122B-A10B 的自定义量化推理入口，已在单张 32GB RTX 5090 上运行。模型使用三值专家、非专家线性层与 embedding 4bit、rank-8 补偿。它是每 token 激活约 10B 参数的 MoE；122B 训练只更新补偿张量，不是全专家权重 QAT。

**先确认手里有指定的 packed 权重。仓库不附权重，原始 BF16 模型不能直接运行这些命令。** 没有产物时，可以先读 [结果汇总](../docs/RESULTS.md) 和 [122B 课程](../courses/122b/README.md)。

2026-10-10，研究已暂停，GPU 与等待任务已释放；代码和证据保留。当前量化质量仍低于原生 BF16 的四组开发评测基线，HTTP 严格输出一致性也尚未通过。可运行的实验入口不等于生产服务验收，状态以 [暂停记录](../docs/experiments/2026-10-10-1150-research-paused.json) 为准。

## 运行前准备

| 项目 | 已验证配置 / 要求 |
|---|---|
| 系统与硬件 | Linux，单张 32GB RTX 5090；无 CPU 权重卸载 |
| Python / PyTorch | Python 3.12，Torch `2.11.0+cu130` |
| 运行库 | Transformers `5.12.1`、Triton；HTTP / vLLM 入口另需准确的 vLLM `0.24.0` 源码 |
| 卷积后端 | 原环境的 `causal_conv1d` CUDA 扩展；替换后端可能改变计算 |
| 模型目录 | 所选版本的完整 packed 产物：配置、tokenizer、`summary.json`、`retained.pt`、embedding 和全部专家 / dense bank 文件 |
| 输出位置 | 从仓库根目录运行；每次选择新的输出文件或目录，不覆盖已有结果 |

HTTP 与下面的新版 CLI 使用 `native4095`，其 manifest（`summary.json`）SHA256 为：

```text
dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1
```

`infer.py` 未指定版本时仍默认使用 `old`，旧版 manifest SHA256 为：

```text
5c9683297d71c1dbdd1c32a8a749aa1911c857fb9ef1a788dfe794230abd08a9
```

入口会校验产物文件和对应源码清单；部分 Transformers 源码也有 SHA 校验，版本号相同不一定足够。本目录不会自动安装 CUDA 扩展或修改系统包。目前尚无从原始 122B 权重生成完整指定产物的一键训练流程。

## HTTP服务入口：完成短程验证

默认监听本机 `127.0.0.1:8000`，保留原预填充和默认输出层。配置为最多 4 条活动序列、256 预填充批次 token、1536 总上下文、896 MiB 缓存。先使用已测的短请求范围：输入不超过 128 token，输出上限 64。

```bash
python experimental122/serve_vllm.py \
  --model /path/to/native4095-packed --output http-original
```

另开终端，调用 OpenAI 兼容的流式聊天接口：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-native122","messages":[{"role":"user","content":"你好，请简短介绍自己。"}],"temperature":0,"max_tokens":64,"stream":true}'
```

服务通过仓库内插件注册模型，主进程与引擎进程核验 68 份源码，不修改共享 `site-packages`。启动会生成 `HTTP-source-policy.json`，加载模型会生成 `weight-proof.json`，可与 [HTTP 来源清单](vllm-http-provenance.json) 核对。

所有请求结束后的 SIGTERM 停服已验证，默认 shutdown timeout 为 30 秒；请求进行中的停服和持续服务尚未验收。较长队列中，两种部署都出现过与旧 CLI 不同的中文措辞。观察性吞吐测量完成不代表严格输出检查通过，详见 [结果汇总](../docs/RESULTS.md) 和 [HTTP 四轮记录](../docs/experiments/2026-10-10-1145-HTTP-observed-ABBA.json)。

### 可选：分组预填充

停掉前一个服务、确认显卡空闲后，使用新输出目录：

```bash
python experimental122/serve_vllm.py \
  --model /path/to/native4095-packed --output http-grouped --grouped-prefill
```

`--grouped-prefill` 显式开启每 16 个专家的分组预填充，默认关闭。它保留相同量化权重，改变 BF16 运算分组，其他输入可能出现舍入或输出变化。短请求回归通过的范围不能推为任意输入逐 token 等价；严格 HTTP 对照失败的结果仍保留。分组预填充与 FP32 输出层是两个独立选项，不能合并各自观测的收益。

## 复现输出层FP32诊断

这个可选候选保持 BF16 压缩权重与持久张量，让输出层 GEMM 返回 FP32，诊断最终舍入造成的并列概率。修正后的服务检查中，热身 8 条和正式 64 条的输出 ID 与旧 CLI 固定样本相同；尚未获得完整的四轮收益结论，默认部署不切换。

准备工具只复制源码，不启动引擎或修改默认入口。生成的 72 份文件与已测候选逐字节一致；同样要求上述 Linux 环境和指定 packed 产物：

```bash
python experimental122/prepare_head_precision_probe.py --output head-probe

BONSAI_HEAD_FP32=1 \
BONSAI_HEAD_CALL_PROOF="$PWD/head-probe/head-call-proof.json" \
python head-probe/experimental122/serve_vllm.py \
  --model /path/to/native4095-packed --output http-head-fp32
```

仍用上面的 curl 调用。必须设置 `BONSAI_HEAD_FP32=1` 才会启用候选；`head-probe/head-call-proof.json` 记录服务内实际输出 dtype，出现不同请求批次后才包含对应行数。单测或插件注册成功不能替代这份调用证据。见 [准备工具](prepare_head_precision_probe.py)、[来源清单](head-precision-provenance.json) 和 [GPU 完成记录](../docs/experiments/2026-10-10-1120-bound-FP32-head-completed.json)。

## 新版 native4095：仓库入口重新验证

不需要 HTTP 时，可用原生单请求入口；明确指定版本，避免加载默认旧版：

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/infer.py \
  --model models/qwen35-122b-native4095-packed --artifact-version native4095 \
  --prompt '用中文解释什么是模型量化。' --max-new-tokens 128 \
  --output outputs/122b-native4095-generation.json

CUDA_VISIBLE_DEVICES=0 python experimental122/infer.py \
  --model models/qwen35-122b-native4095-packed --artifact-version native4095 \
  --benchmark --output outputs/122b-native4095-benchmark.json
```

自然生成按 EOS 停止；`--benchmark` 固定解码 64 步，排除预填充和 Graph 录制，不能与 HTTP 合计吞吐直接比较。五档上下文的 Graph / 静态输出对照见 [验证记录](native4095-verified-run.json)；它们不证明长上下文质量已经恢复。

## 其他入口与评测

| 想做什么 | 从这里开始 |
|---|---|
| 常驻模型、串行请求或缓存容量实验 | [原生请求命令](HISTORY.md#常驻模型与跨请求-graph-复用) |
| vLLM 四请求队列、可选分组预填充 | [B4 缓存组合入口](HISTORY.md#b4-与短上下文缓存的组合完整输出检查完成)、[分组入口](HISTORY.md#分组预填充独立启动入口复刻验证完成) |
| 比较两次运行的完整输出 ID 与 EOS | [CPU 比较器与范围](HISTORY.md#跨引擎的输出比较cpu) |
| 自编短题回归、格式失败样本 | [32 种短请求与评分命令](HISTORY.md#新增32种短请求回归完成保留格式失败样本) |
| 较长输入与隔离测试副本 | [上下文检查命令](HISTORY.md#四条较长输入的分块预填充检查完成)；公开默认 CLI 输入限制仍为 128 |
| 量化质量、吞吐和显存的完整口径 | [结果汇总](../docs/RESULTS.md)、[训练与四组质量记录](native4095-training-result.json) |
| 追查一次实验、失败或源码身份 | [历史记录](HISTORY.md)、[实验索引](../docs/experiments/README.md) |

目录保留各阶段入口与 provenance 清单，方便对照实际实验源码。不要直接合并或重命名 runtime 模块：入口 SHA 校验和旧结果依赖这些文件身份。旧命令、失败样本和当时的限制均收录于历史记录；阶段性的“待验证”不表示目前仍在运行。

<details>
<summary>旧章节链接导航</summary>

以下锚点用于兼容已有文档引用，对应内容已移到历史记录。

<a id="新一轮训练记录2026-10-09"></a>
- [新一轮训练记录（2026-10-09）](HISTORY.md#新一轮训练记录2026-10-09)
<a id="文件与运行环境"></a>
- [文件与运行环境](HISTORY.md#文件与运行环境)
<a id="已有实测"></a>
- [已有实测](HISTORY.md#已有实测)
<a id="常驻模型与跨请求-graph-复用"></a>
- [常驻模型与跨请求 Graph 复用](HISTORY.md#常驻模型与跨请求-graph-复用)
<a id="按请求选择缓存容量与释放空闲块"></a>
- [按请求选择缓存容量与释放空闲块](HISTORY.md#按请求选择缓存容量与释放空闲块)
<a id="按块生成长输入因果掩码的独立实验"></a>
- [按块生成长输入因果掩码的独立实验](HISTORY.md#按块生成长输入因果掩码的独立实验)
<a id="录制-graph-前释放空闲预填充块"></a>
- [录制 Graph 前释放空闲预填充块](HISTORY.md#录制-graph-前释放空闲预填充块)
<a id="启动参数调优的完整反例"></a>
- [启动参数调优的完整反例](HISTORY.md#启动参数调优的完整反例)
<a id="不同长度请求的微批处理进展"></a>
- [不同长度请求的微批处理进展](HISTORY.md#不同长度请求的微批处理进展)
<a id="跨批次-graph-复用入口与端到端反例"></a>
- [跨批次 Graph 复用入口与端到端反例](HISTORY.md#跨批次-graph-复用入口与端到端反例)
<a id="b4-共享输出头32-请求对照完成"></a>
- [B4 共享输出头：32 请求对照完成](HISTORY.md#b4-共享输出头32-请求对照完成)
<a id="共享输出头的独立吞吐对照完成"></a>
- [共享输出头的独立吞吐对照完成](HISTORY.md#共享输出头的独立吞吐对照完成)
<a id="两项预填充优化的完整-abba-对照"></a>
- [两项预填充优化的完整 ABBA 对照](HISTORY.md#两项预填充优化的完整-abba-对照)
<a id="低秩修正批量化候选"></a>
- [低秩修正批量化候选](HISTORY.md#低秩修正批量化候选)
<a id="低秩修正-bmm完整吞吐对照完成"></a>
- [低秩修正 BMM：完整吞吐对照完成](HISTORY.md#低秩修正-bmm完整吞吐对照完成)
<a id="低秩-bmm-的-b2b4-与单请求收尾检查完成"></a>
- [低秩 BMM 的 B2/B4 与单请求收尾检查完成](HISTORY.md#低秩-bmm-的-b2b4-与单请求收尾检查完成)
<a id="精确查找表解包数值与单请求吞吐对照完成"></a>
- [精确查找表解包：数值与单请求吞吐对照完成](HISTORY.md#精确查找表解包数值与单请求吞吐对照完成)
<a id="原串行与优化-b4整套部署对照完成"></a>
- [原串行与优化 B4：整套部署对照完成](HISTORY.md#原串行与优化-b4整套部署对照完成)
<a id="原串行与优化-b4全程显存采样完成"></a>
- [原串行与优化 B4：全程显存采样完成](HISTORY.md#原串行与优化-b4全程显存采样完成)
<a id="在已有优化-b4-上叠加查表完整对照完成"></a>
- [在已有优化 B4 上叠加查表：完整对照完成](HISTORY.md#在已有优化-b4-上叠加查表完整对照完成)
<a id="vllm-denseembedding-适配组件验证完成"></a>
- [vLLM dense/embedding 适配：组件验证完成](HISTORY.md#vllm-denseembedding-适配组件验证完成)
<a id="vllm-完整引擎短请求与图回放验证完成"></a>
- [vLLM 完整引擎：短请求与图回放验证完成](HISTORY.md#vllm-完整引擎短请求与图回放验证完成)
<a id="跨引擎的输出比较cpu"></a>
- [跨引擎的输出比较（CPU）](HISTORY.md#跨引擎的输出比较cpu)
<a id="原生与完整-vllm-图模式四轮吞吐及全程显存对照完成"></a>
- [原生与完整 vLLM 图模式：四轮吞吐及全程显存对照完成](HISTORY.md#原生与完整-vllm-图模式四轮吞吐及全程显存对照完成)
<a id="vllm-四请求并发与每请求预填充数值验证完成"></a>
- [vLLM 四请求并发与每请求预填充：数值验证完成](HISTORY.md#vllm-四请求并发与每请求预填充数值验证完成)
<a id="vllm-文本窗口位置缓存独立入口验证完成"></a>
- [vLLM 文本窗口位置缓存：独立入口验证完成](HISTORY.md#vllm-文本窗口位置缓存独立入口验证完成)
<a id="b4-原生与-vllm-请求队列四轮对照完成"></a>
- [B4 原生与 vLLM 请求队列：四轮对照完成](HISTORY.md#b4-原生与-vllm-请求队列四轮对照完成)
<a id="vllm-短上下文缓存四轮对照完成"></a>
- [vLLM 短上下文缓存：四轮对照完成](HISTORY.md#vllm-短上下文缓存四轮对照完成)
<a id="b4-与短上下文缓存的组合完整输出检查完成"></a>
- [B4 与短上下文缓存的组合：完整输出检查完成](HISTORY.md#b4-与短上下文缓存的组合完整输出检查完成)
<a id="b4-原入口与短上下文缓存组合四轮对照完成"></a>
- [B4 原入口与短上下文缓存组合：四轮对照完成](HISTORY.md#b4-原入口与短上下文缓存组合四轮对照完成)
<a id="b4组合入口缓存1gib与896mib四轮对照完成"></a>
- [B4组合入口缓存1GiB与896MiB：四轮对照完成](HISTORY.md#b4组合入口缓存1gib与896mib四轮对照完成)
<a id="缓存边界与cuda轨迹完成保留负结果"></a>
- [缓存边界与CUDA轨迹：完成，保留负结果](HISTORY.md#缓存边界与cuda轨迹完成保留负结果)
<a id="每16个专家分组预填充同缓存四轮对照完成"></a>
- [每16个专家分组预填充：同缓存四轮对照完成](HISTORY.md#每16个专家分组预填充同缓存四轮对照完成)
<a id="分组预填充独立启动入口复刻验证完成"></a>
- [分组预填充独立启动入口：复刻验证完成](HISTORY.md#分组预填充独立启动入口复刻验证完成)
<a id="新增32种短请求回归完成保留格式失败样本"></a>
- [新增32种短请求回归：完成，保留格式失败样本](HISTORY.md#新增32种短请求回归完成保留格式失败样本)
<a id="四条较长输入的分块预填充检查完成"></a>
- [四条较长输入的分块预填充检查：完成](HISTORY.md#四条较长输入的分块预填充检查完成)
<a id="较长输入的32请求四轮吞吐对照完成"></a>
- [较长输入的32请求四轮吞吐对照：完成](HISTORY.md#较长输入的32请求四轮吞吐对照完成)
<a id="热身后-http-四轮吞吐测量完成"></a>
- [热身后 HTTP 四轮吞吐：测量完成](HISTORY.md#热身后-http-四轮吞吐测量完成)

</details>
