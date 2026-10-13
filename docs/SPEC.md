# SPEC.md — MUM-Gravity-0.1 先导版实现规格（单一事实来源）

> 依据：《技术方案 v2.0》+《评审修订建议 v2.1》。本规格是实现的唯一合同，子代理必须严格遵守接口与边界。
> **环境偏离声明（已确认）**：无网络，无法安装 REBOUND / pytest / safetensors。
> - 教师积分器改为**自写高精度积分器**（`integrators.py`，numpy，float64），二体由 **Kepler 解析解**担任绝对裁判；此偏离写入 RESULTS.md。
> - 测试用纯 Python 断言脚本（`python tests/test_xxx.py` 可直接运行，全过打印 PASS）。
> - 权重保存用 `torch.save` `.pt` 文件（safetensors 不可用，RESULTS.md 注明）。

## 0. 硬约束（违反即返工）

1. **纯模型推理路径**：`inference/` 与 `model/` 的 rollout 不 import 任何教师积分器/求解器；模型只凭权重 + 归一化配置推理。教师代码只允许出现在 `data/`（生成标签）与 `evaluation/`（作参考对照）。
2. **归一化**：一次 rollout 内基准固定（M0, R0, V0, L0, T0），禁止每步重新缩放。
3. **对称成对力**：`F_ij` 沿 `d_ij` 方向、标量强度 `s_ij = softplus(MLP(z_ij))`、反向等量分配给两端 → 动量/角动量在浮点精度内守恒。不宣称能量守恒。
4. **所有 RESULTS.md 数字必须来自真实运行**；失败与超界行为如实记录，禁止编造。
5. **教师数据 float64 存储，模型 float32 训练**。
6. 时间预算：数据生成 < 10 min；训练全程 CPU < 45 min。超出则缩小数据量/步数并在 RESULTS.md 说明实际规模。

## 1. 仓库结构

```
momo-universe-model/
  SPEC.md                        # 本文件的副本引用（主仓库根有一份）
  README.md
  model/
    __init__.py
    normalization.py             # NormState 数据类 + compute_norm / to_norm / from_norm
    encoders.py                  # 节点/边特征构造（不变量标量）
    message_passing.py           # 消息传递层（排列不变聚合）
    pair_force_head.py           # z_ij -> s_ij -> F_ij 对称力头
    gravity.py                   # MUMGravityModel（encode/predict_acceleration/step/rollout）
    state_update.py              # kick_drift_kick
  data/
    __init__.py
    integrators.py               # newton_acc, kdk_leapfrog, rk4, kepler_two_body
    generators.py                # 初态采样（2/3 体）+ 轨迹生成 + npz/manifest 落盘
    splits.py                    # 按 trajectory seed 划分 train/val/test
  training/
    __init__.py
    losses.py                    # L_force / L_state / L_rollout / L_energy（归一化）
    train.py                     # 分阶段训练 CLI：python -m training.train --config configs/pilot.yaml
  inference/
    __init__.py
    load.py                      # 加载权重+配置
    rollout.py                   # strict 模式 rollout：范围检查、停止原因记录
  evaluation/
    __init__.py
    run_eval.py                  # 总入口：python -m evaluation.run_eval 输出全部指标并写 RESULTS 草稿
    symmetry_tests.py            # 排列/旋转/平移一致性
    two_body_check.py            # Kepler 解析对照
    conservation.py              # 动量/角动量/能量漂移
    large_dt.py                  # 大步长判决实验
  configs/
    pilot.yaml                   # 若无 yaml 库则用 configs/pilot.json（二选一，以环境可用为准）
  tests/
    test_tensors.py              # 形状、padding、mask 合同
    test_symmetry.py             # 力头对称性快速单测
    test_integrators.py          # 自写积分器：二体守恒量 + Kepler 对照（教师质量验收）
  results/                       # 运行产物（npz、曲线 png、metrics json）
  checkpoints/                   # 权重 .pt + config json
  RESULTS.md                     # 真实结果报告（成功与失败都写）
```

## 2. 数据合同

### 2.1 归一化（对应 v2.0 §5.2，固定基准）

```
M0 = Σ m_i ;  R0 = Σ m_i r_i / M0 ;  V0 = Σ m_i v_i / M0
L0 = 质量加权 RMS 半径（初态，>0 检查）
T0 = sqrt(L0^3 / (G*M0)), G=1 ;  V_scale = L0/T0
x_i = (r_i − R0 − V0·t)/L0 ;  u_i = (v_i − V0)/V_scale ;  μ_i = m_i/M0
```

`NormState` 保存 M0,R0,V0,L0,T0,V_scale；提供 `to_norm(state,t)` / `from_norm(x,u,t)` 往返（往返误差 < 1e-12 相对）。

### 2.2 教师积分器（data/integrators.py，全部 float64）

- `newton_acc(mass[N], pos[N,3]) -> acc[N,3]`：O(N²) 解析牛顿加速度。
- `kdk_leapfrog(acc_fn, mass, r0, v0, dt, steps) -> (t[steps+1], pos, vel)`。
- `rk4(acc_fn, mass, r0, v0, dt, steps) -> (t, pos, vel)`：教师主力；用更小 dt。
- `kepler_two_body(m1, m2, r0, v0, t_array) -> (pos[2,T,3], vel[2,T,3])`：由能量/角动量定轨道要素，偏近点角用 Newton 迭代解 Kepler 方程（椭圆 E<1）与双曲情形；圆/近圆退化处理。**这是二体的绝对裁判。**
- 教师质量验收（test_integrators.py）：二体 KDK/RK4 对照 Kepler，RK4 在 dt=0.001（归一化时间）下 256 步位置误差 < 1e-6；能量/角动量漂移 < 1e-8（RK4）。

### 2.3 初态采样与轨迹生成（data/generators.py）

归一化单位内直接在"归一化坐标"采样（G=M0=L0=1）：

- **二体**：圆轨道与椭圆轨道（e ∈ [0, 0.8]），质量比 ∈ [1, 100]，随机取向。数量：**400 条**（train 280 / val 60 / test 60，按 seed 划分）。
- **三体**：质量比 ∈ [1, 10]（先导收窄自 v2.0 的 100:1，RESULTS 注明），位置在单位球内、速度缩放到大致束缚（负总能量）且排除过近初始对（最小初始间距 > 0.1）；含 10% 近遇压力样本桶（最小初始间距 ∈ [0.05, 0.1]）。数量：**800 条**（train 600 / val 100 / test 100）。
- 每条轨迹：RK4 dt=0.002 积分，按输出间隔 0.01（归一化时间）保存 **128 帧**（含 t=0），总时长 1.27。每帧存 pos/vel/解析 acc 标签。非有限值或最小间距 < 0.02 时截断该轨迹并记 `truncated=true` 与原因。
- 落盘：`results/data/two_body_{split}.npz`、`three_body_{split}.npz` + `manifest.json`（seed、版本、dt、覆盖统计）。

### 2.4 张量合同（模型输入，float32）

| 字段 | 形状 | 说明 |
|---|---|---|
| mass/μ | [B,N,1] | 归一化质量，padding 为 0 |
| pos | [B,N,3] | 归一化位置 |
| vel | [B,N,3] | 归一化速度 |
| body_mask | [B,N] bool | 有效天体 |
| acc 标签 | [B,N,3] | 解析牛顿加速度（归一化单位） |

## 3. 模型合同（对应 v2.0 §6）

```python
class MUMGravityModel(nn.Module):
    def __init__(self, hidden_dim=128, message_layers=4): ...
    def predict_acceleration(self, mu[B,N,1], pos[B,N,3], mask[B,N]) -> acc[B,N,3]
        # 节点标量编码 -> message_layers 层消息传递（mean 聚合，mask 不计入）-> 不变量 h_i
        # 无序对 i<j：z_ij=[h_i+h_j, |h_i−h_j|, μ_i+μ_j, μ_iμ_j, log r_ij]
        # s_ij = softplus(MLP(z_ij)) ; F_ij = s_ij * d_ij / r_ij ; a_i += F_ij/μ_i ; a_j −= F_ij/μ_j
        # r_ij 下限 clamp 1e-4；padding/对角线全部屏蔽；返回 float32
    def step(self, mu, pos, vel, mask, dt) -> (pos_next, vel_next)   # kick-drift-kick
    def rollout(self, mu, pos0, vel0, mask, dt, steps, limits) -> RolloutResult
        # RolloutResult: pos[steps+1], vel, t, stopped(bool), stop_reason(str|None), min_separation 序列
```

- `limits`: dict(max_steps, min_separation=0.02, max_radius=10.0)；超界停止并记录原因，不发散到 NaN。
- 参数量打印并写入 checkpoints/config.json。

## 4. 训练合同（training/）

阶段课程（每阶段独立的 epoch 上限，loss 权重按阶段切换）：

| 阶段 | 数据 | 损失 | epoch 上限 |
|---|---|---|---|
| S1 force | 二体+三体 train 帧 | L_force | 20 |
| S2 one-step | 同上 | L_force + L_state | 20 |
| S3 rollout | 三体 train 轨迹 | L_force + L_state + L_rollout(4→8 步) + 0.01·L_energy | 30 |

- 优化器 AdamW lr=3e-4, weight_decay=1e-5, grad clip 1.0, batch=32 条轨迹（或等价帧批），seed=42。
- 损失全部按参考尺度归一化（力按帧内 RMS 加速度，状态按 L0/V_scale），同时记录绝对值。
- 每阶段结束存 `checkpoints/mum_gravity_pilot_stage{S}.pt` + config；用 val 集选最佳 checkpoint，**test 只在最终评测用一次**。
- 若 CPU 太慢：优先砍 epoch（如实记录实际 epoch 数），不得砍测试协议。

## 5. 评测合同（evaluation/，全部真实运行，输出 `results/metrics.json` + PNG）

1. **对称性测试**（symmetry_tests）：排列/随机旋转/平移输入 → 预测加速度相应变换后误差 < 1e-5 相对；动量守恒 |ΣF| < 1e-6。
2. **二体 Kepler 对照**（two_body_check）：test 集圆/椭圆各若干条，模型 rollout（dt=0.01）vs Kepler 解析：报告 128 帧的相对位置误差中位数与末帧误差；同图对照"解析牛顿力+KDK(dt=0.01)"基线。
3. **三体短窗**（run_eval 内）：test 集 4/8/16 步 rollout 误差曲线（中位数 + 90 分位）；动量/角动量/能量漂移（conservation）。
4. **大步长判决实验**（large_dt）：在 test 轨迹上比较三条路径的 16 步（物理时长 0.8）末端误差——(a) 解析牛顿力+KDK，dt ∈ {0.01, 0.05}；(b) 模型 rollout，dt ∈ {0.01, 0.05}；(c) 参考 RK4 dt=0.001 作为真值。判决问题：**模型在 dt=0.05 是否优于解析力在 dt=0.05**（若是 → 大步长故事初步成立；若否 → 如实报告）。
5. **独立性验收**：`inference/rollout.py` 在禁止 import `data/` 的前提下完成 rollout（用一个 assert 脚本静态检查 import 图即可）。

## 6. RESULTS.md 必须包含

- 环境偏离声明（无 REBOUND、自写积分器、教师质量验收数字）
- 数据规模实况（轨迹数、帧数、截断率）
- 参数量、训练实际耗时/epoch
- 全部评测数字 + 失败案例（发散轨迹比例、最坏样本）
- 大步长判决结论（成立/不成立，附数字）
- 已知限制清单（长时混沌未验证、近遇软化未做、N>4 未训练等）

## 7. 完成定义（DoD）

- [ ] tests/ 三个测试脚本全 PASS
- [ ] 二体 Kepler 对照通过（RK4 教师验收误差达标；模型误差有真实数字，不要求超越解析）
- [ ] 三体短窗训练收敛（val loss 下降曲线落盘），模型 rollout 无 NaN
- [ ] 大步长实验有明确结论数字
- [ ] RESULTS.md 完整，无任何编造数字
- [ ] git 提交：至少 3 个语义化 commit（core model / data+training / evaluation+results）
