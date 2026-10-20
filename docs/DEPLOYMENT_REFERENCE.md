# 部署参数与排障

首次运行按[快速部署](DEPLOYMENT.md)。这里用于调上下文、测速度和定位报错。

## 历史测量怎么读

| 产物 | 128 token | 4K | 16K | PyTorch峰值 allocated |
|---|---:|---:|---:|---|
| 35B-8192 | 59.33 token/s | 49.85 token/s | 32.71 token/s | 约14.85 / 15.61 / 18.26 GiB |
| 122B-native4095 | 28.75 token/s | 25.84 token/s | 19.01 token/s | 16K约30.26 GiB |

以上是历史固定64步Graph热解码，即使遇EOS也继续；不含加载、编译、预填充或Graph录制，不能当作端到端聊天速度、首token延迟或并发吞吐。显存列是PyTorch allocated，并非整卡NVML占用。35B新便携入口尚未GPU复验，不能把历史结果当成新入口已测。

27B公开32步版已完成便携benchmark四轮八题对照：参考约0.69、仅直接解包3.46、加入融合旋转 **9.37 token/s**，首token计算时间中位数0.305秒，峰值allocated从24.69降至8.12GiB；输出与每步logits一致。未发布96步候选另测为9.74 token/s。两者都是短提示热计算计时，质量仍未通过，优化继续进行；[详细口径与复测](../courses/qwen38_bonsai2/inference/README.md)。

原始结果：[35B五档测量](experiments/2026-10-09-2156.json) · [推理记录](experiments/INFERENCE.md) · [质量与速度口径](RESULTS.md)。

## 35B配置与测速

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


## 122B配置与测速

公开包只用 `experimental122/infer_hf.py`。旧HTTP服务固定原始清单，不能直接加载脱敏公开包；历史四路23.30 token/s属于合计吞吐，且存在输出差异。

```bash
CUDA_VISIBLE_DEVICES=0 python experimental122/infer_hf.py \
  --model models/r122/packed --benchmark --contexts 128 512 4096 \
  --output outputs/r122-graph-benchmark.json
```

先短后长。约28.10GiB常驻文本权重不会随上下文缩短而减少；16K只余很少显存，不建议与桌面或其他模型共卡。历史HTTP与诊断入口见[旧版实验指南](../experimental122/LEGACY.md)。

## 环境与复验

记录组合：Python3.12、Torch2.11.0+cu130、Transformers5.12.1、Triton3.6.0、causal-conv1d1.7.0（27B不要求卷积扩展）。版本固定不等于完整容器锁；驱动和编译器需自行准备。

参考[PyTorch安装表](https://pytorch.org/get-started/previous-versions/)和[卷积扩展源码](https://github.com/Dao-AILab/causal-conv1d)。环境检查器仅报告版本/NVML，不初始化CUDA。27B共享加载/生成代码已由便携benchmark覆盖，单独交互CLI未直接GPU调用；35B便携入口仍待GPU复验。

下载时保留完整目录，尤其 `packed/`、tokenizer及发布清单。`--verify-only`通过后再实际生成，核对文本、EOS和资源记录；用自己的中英文与格式要求检查实用性。

## 常见问题

| 现象 | 处理 |
|---|---|
| `Failed to initialize NVML: Driver/library version mismatch` | 先协调修复宿主机驱动；本工具不会重启、卸载驱动或停止桌面会话 |
| `causal_conv1d`缺失、ABI/undefined symbol | 在已装好对应Torch/CUDA工具链的隔离环境重建1.7.0扩展，再检查；不把静默回退当成历史同路径 |
| Transformers/Triton校验或API错误 | 回到记录版本；不同版本是否可用需另测，不删除断言绕过 |
| OOM或Graph录制失败 | 检查其他进程及整卡空闲，先用短上下文、单请求；不要只看packed文件体积 |
| `Expected ... release`或SHA不符 | 检查模型/标签/路径；35B、122B、27B入口不能互换，脱敏清单与旧清单有意不同 |
| 重复使用`--output`路径 | 每次使用新文件名；35B/27B会拒绝已有路径，122B可能覆盖旧结果 |
| 首次很慢 | 下载、权重加载、Triton编译、预填充和Graph录制都不在历史热解码数字里；优先用常驻35B聊天模式 |
