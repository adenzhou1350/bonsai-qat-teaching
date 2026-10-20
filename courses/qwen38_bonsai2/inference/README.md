# 27B：让参考推理更快

单张RTX 5090上，**公开32步版**已通过便携benchmark四轮、每轮八条短提示对照。权重不变，生成token和每一步最后位置logits均与参考路径一致。内核此前已通过73项算子检查（66项数值精确、7项非法输入拒绝）。

| 路径 | 首token后解码 token/s | 首token计算中位数 | 峰值allocated |
|---|---:|---:|---:|
| 参考路径，前/后复测 | 0.687 / 0.688 | 1.647 / 1.644秒 | 24.69GiB |
| `decode` | 3.458 | 0.490秒 | 8.12GiB |
| `decode_rotate` | **9.371** | **0.305秒** | **8.12GiB** |

这是batch=1、greedy、动态KV缓存、每题最多32个新token的热计算计时：每题预热最多两个token，首token含预填充；解码计时含forward、argmax和CUDA同步，不含加载、tokenization、logits回传/哈希、EOS处理或输出。速度按八题共130个首token后token除以总计算时间得出，不是端到端聊天吞吐。显存为PyTorch峰值allocated，并非整卡占用。

**质量尚未通过，本轮优化已停止。** 32步版仍混淆量化与检查点，首都题不遵守“只答城市名”；四题达到32-token上限而未EOS。[32步完整证据](../../../docs/experiments/2026-10-20-Q38-step32-fast-portable.json)。未发布96步候选另测为9.738 token/s，质量同样未通过；[96步及算子证据](../../../docs/experiments/2026-10-20-Q38-inference-v1.json)。

后续同32步权重的私有GDN Graph四轮八题对照约 **9.40→13.02 token/s（+38.6%）**，输出/logits/cache一致。便携Graph入口尚未GPU验证，本页命令继续使用公开的9.37路径；[收尾记录](../../../docs/experiments/2026-10-20-q38-inference-closeout.md)列出显存、计时和未采用方案。

## 改了三个地方

1. **直接解包**：Triton把每字节5个三值码直接解到目标精度，减少完整中间张量。
2. **按行读取embedding**：只解输入token需要的行，避免展开整个词表。
3. **融合旋转**：单个kernel完成带符号的1024块Hadamard，阶段内保持FP32运算。

代码见[kernels.py](kernels.py)与[runtime.py](runtime.py)。线性层仍先解到BF16再调用`F.linear`；后续直接压缩GEMV候选未通过整模型数值检查，未采用。

## 下载后自测

先完成[环境安装与32步权重下载](../../../docs/DEPLOYMENT.md#27b)，在仓库根目录运行。benchmark已覆盖下列入口共用的加载/生成代码；单独交互CLI尚未直接GPU调用。

```bash
python courses/qwen38_bonsai2/inference/infer.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer --verify-only
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/inference/infer.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer \
  --mode decode_rotate --prompt '请用一句话介绍北京。' --max-new-tokens 32 \
  --output outputs/q38-fast-first.json
```

<a id="reference"></a>

输入含模板最多128 token，输出1–64 token，遇EOS停止。只暴露一张空闲且已协调使用的卡，输出选择新文件名。可将`--mode`换为`reference`回退，或用`decode`比较仅解包的路径；[原参考入口](../infer_prompt.py)仍保留。

## 复刻四轮对照

```bash
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/inference/test_operators.py \
  --device cuda --output outputs/q38-operators.json
CUDA_VISIBLE_DEVICES=0 python courses/qwen38_bonsai2/inference/benchmark.py \
  --packed models/q38-step32/packed --tokenizer models/q38-step32/tokenizer \
  --output outputs/q38-matched-runs
```

`benchmark.py`固定按reference → decode → decode_rotate → reference跑同八题，逐步核对token与logits哈希，输出到新目录；它就是本页32步结果使用的便携入口。复测96步候选需先取得[续训产物](../GUIDE.md#continue96)并替换`--packed`，其权重尚未发布。
