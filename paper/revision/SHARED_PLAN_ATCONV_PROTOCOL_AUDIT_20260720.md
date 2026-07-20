# Shared plan、Protocol V3 与 ATConv 代码审计

审计日期：2026-07-20  
远端项目：`<PROJECT_ROOT>`  
审计对象：90 单元有序消融、R11 shared plan、R11 no-rewrite、Protocol V3 锁、ATConv 插件与论文表述

## 1. 审计结论

| 问题 | 结论 | 论文处理 |
|---|---|---|
| Shared plan 是否严格遵循原始 V3/LCAugv2 计划？ | **否** | 必须称为在 V3 数据与评估协议上另行冻结的 R11 训练策略修订 |
| 90 单元是否共享同一数据划分、mask 规则和评估器？ | **是** | 可作为同一 data/evaluation protocol 下的回顾性有序消融 |
| 90 单元运行谱系是否完整？ | **结果与 checkpoint/control 谱系完整；日志有 1 个例外** | 90/90 个训练单元和 360/360 个文本控制导出齐全；89 个训练日志完整，1 个中间消融日志发生重复启动混写 |
| R11 与 R11 no-rewrite 是否使用相同图像变换？ | **是** | 可作为“只切换文本改写”的精确机制对照 |
| ATConv 是否一直没有启用？ | **否** | 4 个注册替换中 2 个进入前向图、2 个为 dormant legacy modules |
| 是否可以继续写 ATConv4？ | **不可以** | 只能写 two forward-active projector ATConv replacements |
| 现有 90 单元结果能否用于论文？ | **可以，但必须按实际执行协议重命名和降级归因** | 不需要为 3 天投稿冲刺重跑 90 单元 |

总判定：`PASS_WITH_DISCLOSED_DEVIATIONS_AND_ONE_LOG_EXCEPTION`。主结果、checkpoint 和预测/评估谱系可追溯，但不能宣称 Shared plan 是原始 `MEDSEG_TEXT_V3_20260710` 增强计划的严格实现，不能把注册数量 4 当成有效 ATConv 数量 4，也不能把现有代码包称为训练时源码的 bitwise snapshot。

## 2. Shared plan 与原计划的差异

原计划要求：

- BUSI、BUS-BRA、BRISC 使用 hflip/vflip/rot90。
- ClinicDB、COVID-19 使用 hflip-only。
- 禁止会改变位置但无法精确改写文本的任意角度旋转、平移和随机裁剪。
- 颜色、gamma、噪声等非几何增强可以保留。

实际代码存在两条不同路径：

1. 原始 V3 路由位于 `smoke_tests/augmentation_plugins.py:574-595`。它把 BUSI/BUS-BRA/BRISC 路由到 multi-geometry，把 ClinicDB/COVID-19 路由到 hflip-only，符合原计划。
2. 90 单元中的 R11 shared plan 位于同文件 `:469-480`，无论数据集都强制 `multi_geometry=True` 且 `include_affine_zoom=True`；no-rewrite 路径 `:554-571` 使用相同设置。

因此 R11 实际包含五数据集统一的翻转、直角旋转、轻微任意角度 affine、平移和 zoom。它不是原始 V3 增强路由，而是后续训练策略修订。当前论文已明确区分：

- `protocol_hash` 只代表固定的数据注册表、manifest、split、mask 与评估规范；evaluator 源码身份由独立的 `code_sha256` 记录；
- R11 是 separately frozen follow-up amendment；
- 90 单元属于 retrospective ordered ablation，不是原始 7 月 10 日增强规范下的前瞻性 confirmatory matrix。

另一个容易误解的点是：`V3_ABL_BASE` 也使用 `lcaug_v2_hflip_dataset`，并非完全无增强的裸模型。因此论文中的 “Strict Base” 已改为 “horizontal-flip Base reference”。

## 3. 90 单元运行谱系

自动审计脚本：`paper/revision/audit_protocol_v3_factorial_run_lineage.py`  
深度审计输出：`paper/results/bmc_submission_audit_20260719/factorial_run_lineage_deep.json`

结果：

- 6 configurations × 5 datasets × 3 seeds = 90/90 个训练单元齐全。
- 90 × 4 inference controls = 360/360 个 true/shuffled/fixed/empty 导出齐全。
- 90 个唯一 checkpoint SHA256。
- 90/90 个 checkpoint 文件已现场重算 SHA256，全部与 `run_meta.json` 匹配。
- 90 个唯一 run-config SHA256。
- 全矩阵代码哈希一致：`2732eeac3bfcb99163752ad95b1037e36db1fdf3e829b0963a908c2a67ad4b6e`。
- 全矩阵记录的 Git commit 一致：`f3afe35189889bad2f2a3896461deb8d385892cc`。
- 每数据集 manifest 哈希在 18 个配置/种子组合中一致。
- 所有训练命令均核对了 100 epochs、batch size 8、workers 2、split seed 123、cosine scheduler、minimum LR、recipe 和 augmentation 名称。

这些检查证明 90 单元表确实对应同一执行批次和固定数据/评估身份，不是由旧 R7/R8 或不同文本版本拼接而成。

进一步逐日志检查发现 89/90 个 `train.log` 含完整 100 个 epoch，且声明 checkpoint 与最终 `best_checkpoint` 一致。唯一例外是 `V3_ABL_BIOMED_ATCONV/medclipseg_covid19/seed789`：日志只有 44 条可解析的 epoch 记录、包含 7,919 个 NUL 字节，并混入两个 checkpoint 路径，符合重复启动后两个进程写同一文件的特征。其 `run_meta.json` 和四类预测控制均一致绑定 checkpoint `20260717_170631_810637_best.pt`（SHA256 `82e42695...`），且已对该文件重算 SHA256，因此数值仍可反查到唯一 checkpoint，但该训练日志不能称为完整 provenance。

训练时记录的聚合代码哈希为 `2732eeac...`。当前源码、2026-07-15 复现包和 2026-07-18 复现包分别计算为 `ba16f1ee...`、`b97c3da7...` 和 `15279ee6...`，均不匹配。记录的 Git commit `f3afe351...` 又缺少 16 个哈希源文件中的 `run_protocol_v3.py`、`predict_protocol_v3.py`、`evaluate_predictions_v3.py` 和 `protocol_v3/core.py`。所以代码哈希可以识别同一执行批次，但现有材料尚不能 bitwise 恢复训练时的完整源码集合。

## 4. Rewrite/no-rewrite 是否真正配对

`smoke_tests/train_baselines.py:71-76` 的 `augmentation_plan_seed_key()` 会从 no-rewrite 名称中移除 `_no_text_rewrite`。`ManifestDataset.__getitem__()` 在 `:560-582` 使用 `seed + epoch + case_id + normalized plan key` 构造确定性 RNG；训练循环在 `:1728-1729` 每个 epoch 调用 `set_epoch()`。

所以主 R11 与 R11 no-rewrite 在同一 seed、epoch、case 上获得完全相同的图像和 mask 变换，只切换文本改写开关。这个机制对照成立。

例外：R11-LR 的 augmentation key 不同，不能声称逐病例逐 epoch 精确配对。论文已将其降为 same-distribution exploratory diagnostic。

## 5. ATConv 前向激活审计

插件 `smoke_tests/causal_atconv_plugin.py:111-128` 按优先级注册最多 4 个替换。报告 checkpoint 中的注册目标是：

| 目标 | 前向状态 | 实测调用 |
|---|---|---:|
| `base_model.proj.vis.1.0` | active | 1 |
| `base_model.proj_ad.vis.1.0` | active | 1 |
| `base_model.neck_ad.coordconv.1.0` | inactive | 0 |
| `base_model.neck_ad.f4_proj3.0` | inactive | 0 |

静态原因：`repos/CausalCLIPSeg/nets/layers.py:109-116` 实例化了 `f4_proj3` 和 `coordconv`，但 `FPN_AD.forward()` 的实际路径 `:122-153` 没有调用二者。

运行时证据：

- 审计脚本：`paper/revision/audit_atconv_forward_activation.py`
- 输出：`paper/results/protocol_v3_final_controls_20260718/atconv_forward_activation_audit.json`
- 2026-07-20 独立重跑输出：`paper/results/protocol_v3_final_controls_20260718/atconv_forward_activation_audit_rerun_20260720.json`
- 数据：BUSI 真正 test case `medclipseg_busi:102_malignant`
- checkpoint SHA256：`df18874b8e55782401046a79e70be6b118e1f104310e0a0618534cc3ffb10e8d`
- 两个 active projector 输出绝对均值分别为 `0.074325` 和 `0.118632`；两个 neck 目标调用次数均为 0。

所以准确结论是：ATConv **不是一直没启用**，而是有效图中有两个 projector ATConv；另外两个注册模块从未进入输出路径。checkpoint 参数量仍包含 dormant modules，复杂度表必须保留这一披露。

## 6. 协议锁仍有的技术缺口

`smoke_tests/protocol_v3/protocol_lock.yaml` 锁定了 protocol id、数据集、split、mask/evaluation 规则、三种子、100 epochs、阈值和统计规则，但没有锁定完整训练策略。

`smoke_tests/run_protocol_v3.py:168-186` 的 `run_config_sha256` 包含 recipe、augmentation、epochs、代码、manifest 和 cache，但不包含 batch size、workers、model id、optimizer、scheduler 与 minimum LR。实际命令在每个 `train.log` 第一行完整保留，因此本批结果可追溯，但 hash 语义不够完整。

已增强 `audit_protocol_v3_factorial_run_lineage.py`：它现在重算全部 checkpoint SHA256，检查 epoch 连续性、NUL 字节、日志中的 checkpoint 绑定，并核对 60 个 ATConv 指定单元与 30 个 standard-conv 对照。深度审计为零硬失败，同时显式报告上述一个日志例外。它仍没有从逐病例预测重算所有 summary，也不能弥补训练时源码快照缺失，因此不等于 bitwise clean-clone 复现通过。

投稿后续代码建议：

1. 将身份拆成 `data_eval_protocol_id` 与 `training_policy_id`，R11 使用独立 policy hash。
2. 把 model、batch size、workers、optimizer、LR、scheduler、minimum LR、resize mode 和 loss 参数纳入 run-config hash。
3. 在 ATConv 注册后执行 graph-reachability/forward-hook preflight；发现零调用目标就启动失败。
4. 新实验将 `atconv4` 内部 recipe 名改为 `atconv2_projector`；历史 run id 保留用于谱系，不回写旧结果。

## 7. 论文可以与不可以声称的内容

可以声称：

- 在固定 V3 数据与评估协议上，R11 shared policy 相对 horizontal-flip Base reference 提高 Public-5 macro Dice。
- shared policy 解释了几乎全部宏观增益；完整模型相对 Base+plan 的残差为 `+0.04 pp`，95% CI 跨 0。
- case-specific privileged prompts 会改变预测，但证据集中于部分数据集。
- 精确方向改写相对 exact no-rewrite 没有宏观精度收益，并在部分敏感性分析中为负。
- 两个 projector ATConv 在前向图中实际生效。

不可以声称：

- Shared plan 严格遵循原始 Protocol V3 的增强规范。
- 90 单元是原始 V3 promotion gate 下的前瞻性 confirmatory experiment。
- ATConv4 有四个有效算子。
- BioMedCLIP、ATConv 或 rewrite 在 shared policy 之上带来已证明的独立精度增益。
- target-derived public prompts 证明了临床报告理解或部署泛化。

## 8. 最终建议

当前不建议为了 3 天投稿窗口重跑 90 单元。正确处理是保留结果、冻结审计产物、如实命名协议修订，并把论文贡献定位为 shared augmentation attribution 与 privileged-prompt reliability audit。若要求训练日志零例外，只需重跑上述一个 COVID-19 中间消融单元；否则保留其唯一 checkpoint 谱系并披露日志例外。未来版本再修复 hash 语义、源码快照、重复启动锁和 dormant ATConv 注册，并用新 policy id 重跑真正前瞻性验证。
