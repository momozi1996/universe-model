# GitHub 开源备份清单

备份时间：2026-10-13 16:26
备份来源：`/mnt/agents/output/`（校验：备份仓库与原仓库 `diff -rq` 完全一致，git 历史完整）

## 内容

- `momo-universe-model/` — 完整代码仓库（MUM-Gravity-0.1 先导版）
  - 含 `.git` 全部历史（3 个 commit）、模型源码 `model/`、数据生成 `data/`、训练 `training/`、推理 `inference/`、评测 `evaluation/`、测试 `tests/`
  - 含训练权重 `checkpoints/`（3 个阶段 .pt + config）、全部结果 `results/`（metrics.json、图表 PNG、train.log、数据集 npz + manifest）
  - 含 `README.md`、`RESULTS.md`（真实评测报告）、`SPEC.md`（实现规格）
- `docs/` — 开源相关文档
  - `universe-model-idea.md` — 宇宙模型构想 + 第一期选型
  - `mum-gravity-tech-review.md` — 技术方案 v2.0 可行性评审与 v2.1 修订建议
  - `SPEC.md` — 先导版实现规格（单一事实来源）
  - `plan.md` — 实施计划

## git 历史

```
e748aba evaluation + results: full eval suite, real metrics, RESULTS.md, README.md
97bab71 data + training: teacher trajectory generation, staged S1->S3 curriculum
00f2d5c core model: symmetric pair-force GNN, self-written teacher integrators, tensor/symmetry/integrator tests
```
