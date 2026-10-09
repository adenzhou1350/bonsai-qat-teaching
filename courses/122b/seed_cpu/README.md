# 122B 课程：从原始权重构造低比特初始化

这四个 CPU 入口已完成全量运行和独立对照。随附七个源码文件与当时实际执行的文件 SHA 一致，原文件名保留以满足源码自检；见[源码与完成记录](provenance.json)。这一步产出低比特初始化，后续教师生成、专家 rank-8 恢复训练及新产物重载仍未完成。

使用 Linux、Python 3.12.13、Torch 2.11.0+cu130、Transformers 5.12.1、safetensors 0.8.0。全量实验使用这些实际版本；未建立其他版本兼容性。需要原始 Qwen3.5-122B-A10B 的完整 BF16 safetensors 权重、索引及配置/tokenizer 文件，先完成[校准数据课程](../README.md)。无需 GPU，CPU 运行需数小时以上，具体取决于 CPU、内存与存储。

从仓库根目录依次运行，所有输出子目录必须尚不存在：

```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES='' python courses/122b/seed_cpu/portable122_source_ternary_v1.py \
  --model models/Qwen3.5-122B-A10B --output outputs/122b-experts --threads 8
CUDA_VISIBLE_DEVICES='' python courses/122b/seed_cpu/portable122_calibrate_dense4_v1.py \
  --model models/Qwen3.5-122B-A10B --tokens outputs/122b-dense-data/tokens.json \
  --output outputs/122b-dense --layers 48 --threads 8
CUDA_VISIBLE_DEVICES='' python courses/122b/seed_cpu/portable122_embedding4_v1.py \
  --model models/Qwen3.5-122B-A10B --tokens outputs/122b-dense-data/tokens.json \
  --output outputs/122b-embedding
CUDA_VISIBLE_DEVICES='' python courses/122b/seed_cpu/portable122_assemble_seed_v1.py \
  --model models/Qwen3.5-122B-A10B --experts outputs/122b-experts \
  --dense outputs/122b-dense --embedding outputs/122b-embedding --output outputs/122b-seed
```

1. 专家初始化：完整 48 层、96 个 bank、24576 个专家矩阵。按 group128 做 BF16 Lloyd8 三值拟合，五个三值编码放入一个字节；专家梯度更新数为 0，这是 PTQ。
2. Dense 校准：原始 BF16 模型逐层计算隐藏状态，仅 512 个 TRAIN 块更新 Hessian；16 个 VAL 块仅对照。完整 48 层与输出头共 373 个矩阵，输出 affine4/group128。部分 `--layers` 会省略输出头，不能通过完整组装检查。
3. Embedding 初始化：完整 248320×3072 embedding 做 affine4 RTN；固定 128 次 CPU Adam 更新 rank-16 补偿，只有 TRAIN 参与更新，打包基座不变。这与后续专家 rank-8 恢复不同。
4. 组装：校验组件 SHA、原始参数身份及布局，加入 361 个保留张量并复制四个配置/tokenizer 文件，生成 469 个矩阵加 embedding 的完整 seed。见[组装完成证据](../../../docs/experiments/2026-10-10-0218.json)。

Dense 入口检查可用 RAM 大于 24GiB、输出父目录可用磁盘大于 8GiB；这些只是开始运行的门槛，不是完整任务的资源预算。原始 BF16 权重、组件、组装副本与临时激活需要额外存储和内存；组装还按实际组件大小检查空间。代码采用 CPU fallback，未验证 Windows 原生运行。

生成的 `portable-direct-ternary122-dense4-rank8-seed-v1` 尚无专家恢复因子。当前 `experimental122` 推理入口要求另一份已完成 4096 步恢复的固定产物，不能直接加载本课 seed。完成 CPU 初始化不等于重新训练并部署成功；仓库不分发模型权重。

Dense GPTQ 类取自 [jndeng/GEMQ 的固定提交](https://github.com/jndeng/GEMQ/tree/5eb2240cb46d9811bc9f79026100b46f62a7b642)，保留类 AST 与原始运算，见随附 [MIT 许可](portable122-GEMQ-LICENSE.txt)。仅提取 dense GPTQ，不包含 GEMQ 的位宽分配或专家训练。许可正文只将 CRLF 规范为 LF，原始与发布版 SHA 均在 provenance 中记录。
