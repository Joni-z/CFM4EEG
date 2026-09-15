# CroFreMo 模型版本索引

快照 UTC：2026-09-15T05:27:33.940261+00:00

## 使用规则

- 今后明确使用本索引简称，不再单独说“duplex”。简称不表示性能排序。
- 原始配置、run 目录和 checkpoint 名不改；每个简称映射固定的原始 ID。
- 结果必须附数据集、指标、验证/测试、seed 数和结束原因；未完成不填测试分数。
- 新结构先登记结构差异、对照、验证门槛、预算、配置 ID，再提交。完成后保留失败与预算截断记录。
- 下表是两套数据上的版本导航，不是整个项目所有历史实验，也不是 SOTA 排名。
- 同一简称允许数据集必要的损失/类别数差异；完整逐 seed 配置见伴随 JSON。

## 结构和原始 ID

| 简称 | 中文名 | 原始 run 后缀 | 结构 |
|---|---|---|---|
| Duplex-A128 | 早期双路版 | `paclock_duplex` | 8 频段，D128，显式频段 attention；波形与耦合分成两组 token。 |
| Duplex-F192 | 论文双路版 | `cf2_v1d192` | 8 频段，D192，关闭独立频段 mixer，将频段并入空间 attention（space_over_bands）。替换前的论文参考模型，现保留为结构消融。 |
| PAC8 | 8 频段纯耦合版 | `paclock_rot2` | 8 频段，D128，幅度乘以 PAC 对齐的单位相位；没有独立波形 token。 |
| PAC16 | 16 频段纯耦合版 | `crofremo_n5` | 16 频段，D128，纯耦合；n5 配方：wd .05、dropout .35、五种增强。 |
| Quad16 | 正交载波版 | `nb16_quadrature_20260915` | 16 频段，D128，幅度和波形占复数的两个坐标，再作 PAC 旋转；沿用 n5 配方。 |
| Carrier16 | 波形载波版 | `nb16_carrier_20260915` | 16 频段，D128，复数两个坐标都由波形投影得到，再作 PAC 旋转；沿用 n5 配方。 |
| Anchor16 | 有界调制版 | `nb16_anchored_20260915` | 16 频段，D128，波形载波从恒等映射起步，学习有界 PAC 旋转；沿用 n5 配方。 |
| Residual16 | 波形残差版 | `nb16_residual_20260915` | 16 频段，D128，纯耦合 token 加可学习的频段内波形残差；沿用 n5 配方。 |

## 已完成测试结果

均值 ± 样本标准差；单 seed 不写 ±。TUEV 用 κ，CHBMIT 用 PR-AUC。

| 简称 | TUEV κ | CHBMIT PR-AUC |
|---|---:|---:|
| Duplex-A128 | 0.6895 ± 0.0357（3 seeds） | 0.6985 ± 0.0549（3 seeds） |
| Duplex-F192 | 0.6546 ± 0.0365（2 seeds） | 0.6691 ± 0.0413（3 seeds） |
| PAC8 | 0.7328 ± 0.0161（3 seeds） | 0.5060 ± 0.0861（3 seeds） |
| PAC16 | 0.7391（1 seed） | 本次核对未发现该配置完整结果 |
| Quad16 | 0.7279（1 seed） | 训练中，暂无完整测试结果 |
| Carrier16 | 0.7214（1 seed） | 0.6914（1 seed） |
| Anchor16 | 训练中，暂无完整测试结果 | 训练中，暂无完整测试结果 |
| Residual16 | 节点内补位队列，尚未开跑 | 训练中，暂无完整测试结果 |

## 当前判断与在跑任务

- **最新论文主参考已改为 Duplex-A128**，按用户要求统一替换全部12数据集；36份正常完成结果均已核对。Duplex-F192 是替换前的论文版，现保留为结构消融。
- Carrier16 两项主指标的 seed0 数字都高于论文版均值；不能据此宣称稳定领先。TUEV balanced accuracy 只有 .6064，不能说所有指标都强。
- 旧 CHB 验证筛选门槛参照 Duplex-A128，而非论文版。该历史门槛不因查看测试分数而倒改；未达门槛不等于模型没有研究价值。
- 当前 allocation 419232：Quad16/CHB、Anchor16/CHB、Anchor16/TUEV、Residual16/CHB。Residual16/TUEV 在下一张空卡冒烟后补位。
- 实时状态看 results/amd-handover-419232.json、results/backfill-419232.json 和训练 progress.json；本文件是带日期快照。

## 溯源

逐 seed 测试值、选中验证指标、完整配置、原始绝对路径及 SHA256：`MODEL-VERSIONS-2026-09-15.json`。
最新论文映射：`scripts/gen_tables.py` 的主表 CroFreMo → `paclock_duplex`；旧表映射为 `cf2_v1d192`。不手改生成表格。
