# 122B 课程：重建 dense 校准输入

这一课已在 Linux CPU 上实际运行并退出 0：5224 条已批准对话逐字节重建，528 个块、270864 个 token ID 与完整 48 层校准的原输入相同。[完成证据](../../docs/experiments/2026-10-10-0217.json)与[源码身份](provenance.json)单独记录。此入口只准备校准数据；完整教师、恢复训练与新产物重载仍待完成。

使用 Python 3.12、Transformers 5.12.1，以及原始 Qwen3.5-122B-A10B 的完整本地 tokenizer 目录（含 chat template）。CPU 即可，不需要加载模型权重。

```bash
mkdir -p inputs
curl -fL https://raw.githubusercontent.com/sahil280114/codealpaca/2f78ddc5c682ed6738ad092bbbfa59ba915afcb0/data/code_alpaca_20k.json -o inputs/code_alpaca_20k.json
python courses/122b/prepare_dense_data.py \
  --codealpaca inputs/code_alpaca_20k.json \
  --recipe courses/122b/data/approved-recipe.json \
  --tokenizer models/Qwen3.5-122B-A10B \
  --output outputs/122b-dense-data
```

输入源、配方和输出 token 文件均做固定 SHA 检查；目录必须不存在，任何不一致都会拒绝。成功得到 train.json、validation.json、tokens.json 和 summary.json。tokens.json 的 SHA 为 `ce298b506b086b7fecfaaf2b3a25a08895f651ec0a73cc640a0d6a9b44f8c1d9`。配方文件必须保留 LF；仓库属性已固定其换行。

冻结配方包含公开 CodeAlpaca 的行号/身份和自编整数算术，重建 5096 条 TRAIN 与 128 条 VAL 对话。固定种子 113/114 打乱后，拼接完整 user/assistant 对话，截取 512/16 个互不相同的 513-token 块；实际模型输入为每块前 512 位置，不 padding、不重复填充。TRAIN 选中 2113 条代码和 550 条数学，VAL 选中 51 条代码和 49 条数学。只有 TRAIN 更新 Hessian，VAL 仅做重建对照。

这是历史已批准选择的精确重建，不重新运行历史排除过滤，也不证明语义无污染或每个参考回答正确。程序不读取评测/保留文件。后续 rank-8 恢复使用 OpenThoughts 4096/128，是另一份数据。本仓库不分发 CodeAlpaca 原始回答文件或生成的 tokens。

CodeAlpaca 数据源：[sahil280114/codealpaca](https://github.com/sahil280114/codealpaca/tree/2f78ddc5c682ed6738ad092bbbfa59ba915afcb0)，许可为 CC BY-NC 4.0，见随附[完整数据许可](data/CodeAlpaca-DATA_LICENSE.txt)。本仓库代码许可不替代上游数据许可。
