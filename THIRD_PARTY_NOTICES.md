# License scope and third-party notices

The repository author's original code and documentation that they have the right to license are released under the root [MIT License](LICENSE). Third-party material retains its own copyright notices and license terms. The root license does not relicense external model weights, datasets, or dependencies.

## GEMQ dense GPTQ

- Source: [jndeng/GEMQ](https://github.com/jndeng/GEMQ/tree/5eb2240cb46d9811bc9f79026100b46f62a7b642), commit `5eb2240cb46d9811bc9f79026100b46f62a7b642`, `gemq/quantizers/gptq.py`.
- Included component: the dense GPTQ class in [portable122_gptq4_v1.py](courses/122b/seed_cpu/portable122_gptq4_v1.py).
- Copyright (c) 2026 Jianing Deng. The original [MIT notice](courses/122b/seed_cpu/portable122-GEMQ-LICENSE.txt) is retained unchanged.
- The source and extraction identities are recorded in [provenance.json](courses/122b/seed_cpu/provenance.json). This extraction does not include GEMQ bit allocation or expert training.

## CodeAlpaca data

- Source: [sahil280114/codealpaca](https://github.com/sahil280114/codealpaca/tree/2f78ddc5c682ed6738ad092bbbfa59ba915afcb0), commit `2f78ddc5c682ed6738ad092bbbfa59ba915afcb0`.
- The 122B calibration course refers to this external data and keeps the full [CC BY-NC 4.0 data license](courses/122b/data/CodeAlpaca-DATA_LICENSE.txt).
- The original answer file and generated token dataset are not distributed in this repository. The root MIT license does not replace the CodeAlpaca data license; see the [course and fixed recipe](courses/122b/README.md).

## External models, datasets, and libraries

Model weights are not included. The courses and historical experiments refer to external Qwen models, OpenThoughts, WildChat, and other datasets, and use dependencies such as PyTorch, Transformers, Triton, and vLLM. Obtain and use those materials under their respective terms. Their names and links in this repository do not make them part of the root MIT grant.
