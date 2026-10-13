# 不装模型，先核对公开结果

如果你是看到「单张 5090 跑 122B」找来的，可以先花几秒检查这里的数字从哪来。这个入口只需 **Python 3.9+ 标准库**，Windows、Linux 都可以运行；不安装 Torch，不下载权重，也不启动训练。

从仓库根目录执行：

```bash
python scripts/check_public_evidence.py
```

脚本读取仓库里的记录，核对源码哈希，再从四轮时间和 token 总数重算 HTTP 吞吐、请求差异数与显存单位，并检查 R35 / R122 的开发评测计数。正常输出包括：

```text
PASS: public file identities and summary arithmetic only
HTTP pooled output: 21.01 -> 23.30 tokens/s (+10.90%; aggregate of four clients)
HTTP complete requests matching old CLI: 244/256; 12 differ (not a token accuracy score)
HTTP sampled NVML used: 30.64 GiB; minimum free: 740.31 MiB; separate driver reserved: 0.484 GiB
```

**PASS 只说明已公开文件的身份和摘要算术相符。** 它不会重跑模型、检查未发布的逐请求日志，或重新评分。两个量化模型仍有低于各自 BF16 对照的开发评测；HTTP 严格输出一致性仍未通过。计时、分母和限制见[结果说明](RESULTS.md)。

## 固定的是哪一版

[公开证据清单](public-evidence-manifest.json)以提交 [`f6d90b4`](https://github.com/adenzhou1350/bonsai-qat-teaching/tree/f6d90b4332e585737c593c8823516136e1618093) 中的文件为基准，固定 12 份公开清单、实验记录、配方和第三方许可文件；再沿四份已有 provenance 清单核对 85 份去重后的源码及配套文件。文本只做 CRLF → LF 换行归一化。文件缺失、被修改或摘要算术不符时，脚本退出码为 1。

这个基准提交固定已有实验材料；本页、核对脚本和新索引是在其后增加的，不声称当时运行过这个脚本，也不把当前 README 或脚本自身写入哈希形成循环。以后引用新实验，应新增证据快照，保留旧记录。哈希便于比对文件身份，不能单独证明一次实验真实执行或独立复现成功。

## 下载仓库后，实际能拿到什么

| 路线 | 仓库中已经有 | 继续执行还缺什么 |
|---|---|---|
| CPU 教学 | 方程、STE、打包、检查点恢复测试与依赖说明 | Linux CPU 环境与依赖；按[训练指南](REPRODUCING.md)运行。上面的纯标准库检查不代替这些测试 |
| T35 全专家 QAT | 训练、教师缓存、导出和恢复入口 | 原始 35B 模型、自己的文本、2×B300 等资源；原始采样与完整评分链未公开，不能据此复制原分数 |
| R35 / NR35 补偿 | [训练及评测摘要](experiments/35B.md)、产物身份与部分计数 | 独立训练 / 推理入口、指定权重、完整逐题结果与评分流程，当前未随仓库发布 |
| R122 固定产物推理 | [CLI / HTTP 入口](../experimental122/README.md)、源码身份、测量摘要 | 指定 native4095 packed 权重与匹配环境；仓库没有公开下载地址。原始 BF16 权重不能替代 |
| P122 初始化课程 | [校准配方](../courses/122b/README.md)、[CPU 初始化代码](../courses/122b/seed_cpu/README.md)、阶段完成摘要 | 外部原始数据与模型；完整教师、恢复训练、新产物推理链尚未完成。初始化 seed 不能直接交给 R122 入口 |

native4095 指定产物的 `summary.json` SHA256 是 `dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1`。仓库公开了这个身份，**没有公开该权重文件本身**；本脚本也不会据这个字符串宣布权重已经校验。

## 哪些证据能检查到哪一步

| 内容 | 公开文件 | 本次检查范围 |
|---|---|---|
| 教学、HTTP 和 CPU 课程源码 | [教学身份](reproducibility-manifest.json)、[HTTP 身份](../experimental122/vllm-http-provenance.json)、[数据入口](../courses/122b/provenance.json)、[初始化入口](../courses/122b/seed_cpu/provenance.json) | 文件在本地且 SHA 相符；不导入或执行模型代码 |
| R35 开发评测 | [四组计数](experiments/2026-10-09-2245.json) | 计数范围、原有分母与各自 BF16 对照；未重新运行评分 |
| R122 训练与开发评测 | [训练及评测摘要](../experimental122/native4095-training-result.json) | 四组配对通过数、分母与质量标记是否一致；训练损失未重测 |
| R122 HTTP 四轮实验 | [四轮摘要](experiments/2026-10-10-1145-HTTP-observed-ABBA.json) | 合并吞吐重算、244/256 请求相同、采样显存换算；无完整请求、SSE、服务器日志或 NVML 原始采样文件 |
| P122 配方与初始化 | [数据完成摘要](experiments/2026-10-10-0217.json)、[组装摘要](experiments/2026-10-10-0218.json) | 已发布文件的身份；不重建数据、加载模型或重新组装产物 |

这些 JSON 是已发布的实验摘要。里面有些 `summary_sha256`、`terminal_sha256`、`raw_sha256` 指向未随仓库分发的原文件。**有摘要和哈希，不等于读者已经拿到完整原始证据。** 要进一步开放复现，仍需补齐可公开的输入修订、产物取得方式、运行环境、逐题 / 逐请求结果与对应评分或负载脚本。

## 带着结果反馈

如果检查失败，提 Issue 时附上 `git rev-parse HEAD`、Python 版本和脚本输出即可；先不要修改清单里的期望哈希来消除失败。如果已经跑了 Linux CPU 课程，再附实际命令和退出码。这样的记录比单独贴一个性能数字更容易定位问题。

核对脚本自己的标准库测试可以运行：

```bash
python -m unittest discover -s tests -p test_public_evidence.py -v
```

测试会在临时目录模拟证据损坏、源码缺失、换行变化和错误的合并吞吐，不碰原始实验文件。
