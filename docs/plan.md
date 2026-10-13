# plan.md — MUM-Gravity-0.1 代码实现落地

## 目标
按《MOMO Universe Model 技术方案 v2.0》+《评审与修订建议 v2.1》实现可运行的首个模型先导版：
数据生成 → 对称成对力 GNN → KDK 状态更新 → 训练 → 独立推理 → 评测（二体解析裁判、三体短窗、大步长判决实验）。

## 阶段划分

### Stage 1 — 技能加载与基线准备（主代理）
- 加载 `vibecoding-general-swarm`（编码任务强制）
- 探测环境：Python / torch / numpy / 是否能装 rebound（装不上则自写积分器 fallback）
- 建仓库骨架 `/mnt/agents/output/momo-universe-model/`

### Stage 2 — 核心实现（coder 子代理）
按 v2.1 修订后的最小验证路径实现：
1. `model/`：normalization.py（质心共动 + T0 无量纲化）、pair_force_head.py（对称成对力）、gravity.py / message_passing.py（GNN）、state_update.py（kick-drift-kick）
2. `data/`：二体解析数据（Kepler 裁判）+ 三体数值教师数据（REBOUND 或自写 leapfrog/IAS15-lite 高精度积分器）
3. `training/`：losses.py（L_force / L_state / L_rollout / L_energy，分阶段）、train.py（staged curriculum）
4. `inference/`：纯模型 rollout（无求解器依赖）、strict 范围检查
5. `evaluation/`：对称性测试（排列/旋转/平移）、二体解析对照、三体短窗误差、**大步长判决实验**（dt 扫描 vs 解析基线）
6. `tests/`：张量合同、padding、对称性单元测试

### Stage 3 — 训练与验证（coder 子代理，CPU 可行：模型极小）
- 二体先导：解析对照必须过
- 三体短窗训练 + rollout 误差曲线
- 大步长实验：dt ∈ {0.005, 0.02, 0.05} × 精度对照 → 判决模型存在理由

### Stage 4 — 验收与交付（主代理 + reviewer 子代理）
- 对照 §21 验收清单逐项检查（能跑通的部分）
- 交付：代码仓库 + RESULTS.md（真实数字，失败也写）+ REF 标签

## 硬约束（来自 v2.1 评审，子代理必须遵守）
- 纯模型推理路径不得调用求解器/LLM
- 归一化一次 rollout 内固定基准，不得每步重缩放
- 成对力对称构造（动量/角动量精确守恒）；不宣称能量严格守恒
- 所有数字必须来自真实运行；失败案例如实记录
- 评测分"短时族/统计族"两族
