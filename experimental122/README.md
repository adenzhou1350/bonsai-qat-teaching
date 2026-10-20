# 122B：单卡5090推理

Qwen3.5-122B-A10B MoE，三值专家 + 4bit稠密层/embedding + rank8补偿，训练4096步。下载约30.20GB，历史常驻权重约28.10GiB。

**[安装环境与部署](../docs/DEPLOYMENT.md#122b)** · [显存与测速](../docs/DEPLOYMENT_REFERENCE.md) · [评测结果](../docs/RESULTS.md)

当前公开包上传中，标签完成公开后再下载。安装环境后，在仓库根目录执行：

```bash
hf download aden1350/Qwen3.5-122B-A10B-Ternary-Rank8-Experimental \
  --revision native4095-step4096 --local-dir models/r122
python experimental122/infer_hf.py --model models/r122/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python experimental122/infer_hf.py \
  --model models/r122/packed --prompt '请用中文解释模型量化。' \
  --max-new-tokens 128 --output outputs/r122-first.json
```

使用Linux和一张空闲32GB5090；先试短问题，输出用新文件名。公开包须用 **`infer_hf.py`**，旧CLI和HTTP服务不能直接加载它。

历史产物已完成单卡推理；四组开发评测仍低于BF16基线。脱敏公开包通过CPU身份检查，不代表新的GPU复验或质量验收。详情见[发布来源](model-release-provenance.json)。

旧HTTP、vLLM和输出层实验放在[历史入口指南](LEGACY.md)，完整过程见[实验时间线](HISTORY.md)。

<details>
<summary>旧章节链接</summary>

<a id="单张-rtx-5090-跑-122b已验证的实验版"></a>
<a id="新一轮训练记录2026-10-09"></a>
<a id="文件与运行环境"></a>
<a id="已有实测"></a>
<a id="常驻模型与跨请求-graph-复用"></a>
<a id="按请求选择缓存容量与释放空闲块"></a>
<a id="按块生成长输入因果掩码的独立实验"></a>
<a id="录制-graph-前释放空闲预填充块"></a>
<a id="启动参数调优的完整反例"></a>
<a id="不同长度请求的微批处理进展"></a>
<a id="跨批次-graph-复用入口与端到端反例"></a>
<a id="b4-共享输出头32-请求对照完成"></a>
<a id="共享输出头的独立吞吐对照完成"></a>
<a id="两项预填充优化的完整-abba-对照"></a>
<a id="低秩修正批量化候选"></a>
<a id="低秩修正-bmm完整吞吐对照完成"></a>
<a id="低秩-bmm-的-b2b4-与单请求收尾检查完成"></a>
<a id="精确查找表解包数值与单请求吞吐对照完成"></a>
<a id="原串行与优化-b4整套部署对照完成"></a>
<a id="原串行与优化-b4全程显存采样完成"></a>
<a id="在已有优化-b4-上叠加查表完整对照完成"></a>
<a id="vllm-denseembedding-适配组件验证完成"></a>
<a id="vllm-完整引擎短请求与图回放验证完成"></a>
<a id="跨引擎的输出比较cpu"></a>
<a id="原生与完整-vllm-图模式四轮吞吐及全程显存对照完成"></a>
<a id="vllm-四请求并发与每请求预填充数值验证完成"></a>
<a id="vllm-文本窗口位置缓存独立入口验证完成"></a>
<a id="b4-原生与-vllm-请求队列四轮对照完成"></a>
<a id="vllm-短上下文缓存四轮对照完成"></a>
<a id="b4-与短上下文缓存的组合完整输出检查完成"></a>
<a id="b4-原入口与短上下文缓存组合四轮对照完成"></a>
<a id="b4组合入口缓存1gib与896mib四轮对照完成"></a>
<a id="缓存边界与cuda轨迹完成保留负结果"></a>
<a id="每16个专家分组预填充同缓存四轮对照完成"></a>
<a id="分组预填充独立启动入口复刻验证完成"></a>
<a id="新增32种短请求回归完成保留格式失败样本"></a>
<a id="四条较长输入的分块预填充检查完成"></a>
<a id="较长输入的32请求四轮吞吐对照完成"></a>
<a id="热身后-http-四轮吞吐测量完成"></a>
<a id="运行前准备"></a>
<a id="http服务入口完成短程验证"></a>
<a id="可选分组预填充"></a>
<a id="复现输出层fp32诊断"></a>
<a id="新版-native4095仓库入口重新验证"></a>
<a id="其他入口与评测"></a>

对应命令见[历史入口指南](LEGACY.md)与[实验时间线](HISTORY.md)。

</details>
