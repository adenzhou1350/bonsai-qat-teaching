# 35B INT2＋rank16：8192步部署入口

这是Qwen3.5-35B-A3B的R35-8192产物，冻结INT2专家基座、训练rank16补偿，共8192步。不是全专家QAT的T35检查点，也不是27B旋转三值方案。80个专家bank和retained共81个张量文件保持历史字节身份，公开清单只脱敏私有路径和lineage。

**完整安装、下载、单次提问、常驻聊天、4K/16K配置和排障见[部署指南](../docs/DEPLOYMENT.md)。** [来源清单](provenance.json)给出6份原运行时文件的精确SHA，以及便携加载器数值语句的AST对照。新入口完整GPU复验尚待空闲卡；历史权重的独立重载与5090结果见[历史记录](../docs/experiments/2026-10-09-2156.json)。

公开仓库目标为`aden1350/Qwen3.5-35B-A3B-INT2-Rank16-8192-Experimental`，固定标签`recovery-step8192`；目前发布中，完成匿名验证后更新状态。发布完成后从仓库根目录运行：

```bash
hf download aden1350/Qwen3.5-35B-A3B-INT2-Rank16-8192-Experimental \
  --revision recovery-step8192 --local-dir models/r35-8192
python experimental35/infer.py --model models/r35-8192/packed --verify-only
CUDA_VISIBLE_DEVICES=0 python experimental35/infer.py \
  --model models/r35-8192/packed --chat --context 4096 --max-new-tokens 256
```

Linux单张32GB5090，记录环境Python3.12、Torch2.11.0+cu130、Transformers5.12.1、Triton3.6.0、causal-conv1d1.7.0。`--verify-only`无需Torch/CUDA；实际推理需要这些依赖。`/reset`清空历史、`/exit`退出。单进程串行请求，不提供并发线程安全或已验证的HTTP/vLLM服务。

8192步开发成绩为HumanEval141/164、中文/格式48/48、CMMLU158/201、GSM8K190/200；对应原模型129/164、30/48、161/201、192/200。后两项未达基线，不能宣称整体质量恢复。[评测与暂停边界](../docs/experiments/35B.md)保留原始口径。
