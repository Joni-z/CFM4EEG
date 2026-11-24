# AAC 对照：用 AAC 替换 CfFT 中的 PAC

CfFT 用每个 50 点窗口内的 PAC 矩确定 rotation 的参考方向。这个对照只把它换成振幅–振幅耦合（AAC），其余部分都不改：

- PAC：Z_ij = E[(A_j − mean A_j)·e^{iφ_i}] / 100
- AAC：R_ij = E[(A_j − mean A_j)·a_i] / 100，其中 a_i 是源频带包络中心化后缩放到单位 RMS 的结果

两者的源信号都满足 E|source|² = 1，所以 |R_ij| 等于包络 Pearson 相关系数乘以目标包络的标准差，和 |Z_ij| 的尺度一致。R 是实数，原来的对齐代码把 |r| 当权重、把 sign(r) 当方向。参数量（1,224,486）和初始化与 PAC 模型完全相同。

配置为 [tuev-aac-control.yaml](../../configs/model/tuev-aac-control.yaml)，相对 [tuev-scratch.yaml](../../configs/model/tuev-scratch.yaml) 只加了 `coupling_measure: aac`。两臂都用 seed 2，在 Torch a100_tandon（A100-SXM4-80GB）上用同一份代码训练 20 个 epoch。

| | PAC（Torch 重训） | AAC |
|---|---:|---:|
| 测试集 κ | **0.6736** | 0.6421 |
| balanced acc | **0.6374** | 0.5737 |
| weighted F1 | 0.8284 | 0.8146 |
| 最佳验证集 κ（epoch） | 0.6088（3） | 0.5762（4） |
| Torch job | 24096065 | 24097362 |

PAC − AAC：κ +0.0315（95% 区间 [0.0235, 0.0393]），balanced acc +0.0638（[0.0515, 0.0761]）。区间来自测试窗口配对重采样，2000 次，只反映测试样本的不确定性，不包含 seed 方差。这个配置在 amd 上三个 seed 的测试集 κ 标准差是 0.014。

| 召回率 | SPSW | GPED | PLED | EYEM | ARTF | BCKG |
|---|---:|---:|---:|---:|---:|---:|
| PAC | 0.042 | 0.802 | **0.654** | **0.775** | **0.661** | 0.891 |
| AAC | **0.145** | 0.790 | 0.379 | 0.635 | 0.577 | **0.916** |

差距主要在 PLED，其次是 EYEM 和 ARTF。AAC 在 SPSW 和 BCKG 上略高。

## 说明

- 单 seed 结果。
- AAC 第一次运行（job 24096066）在某个包络窗口数值下溢时，sqrt(0) 产生了 NaN 梯度。v2 在开方前对均方值设下限，PAC 路径没有改动；PAC 臂在 v1 上运行。
- Torch 上的 PAC 重训比 9 月 amd 上的同配置 checkpoint（0.7134）低 0.04，原因见 [TUEV 主模型 checkpoint](tuev-main-checkpoint-20261123.md)。AAC 只和同一台机器上的 PAC 重训比较。
- 特征层面的同类比较见 [Power、PAC 与 AAC 的特征增量](pac-power-increment-20261123.md)。

[完整数值、配置与哈希](cfft-aac-control-20261123.json)

## 四个数据集的配对训练（2026-11-24 提交）

四项 AAC 使用 seed 0，各自沿用已完成的双轴 PAC 运行的训练配置，将 `coupling_measure` 设为 `aac`。下表给出提交记录和历史 PAC 测试分数。

| 数据集 | 主指标 | PAC 对照 | PAC 记录 | Torch AAC job |
|---|---|---:|---|---:|
| TUSZ | AUC-PR | 0.6581 | [`3a0d527cd2686f53f7ea`](../datasets/tusz.md) | 24339795 |
| IIIC | κ | 0.4665 | [`5debb59344678e8d4308`](../datasets/iiic.md) | 24339803 |
| ISRUC | κ | 0.7680 | [`bb3b9758768984586dd5`](../datasets/isruc.md) | 24339807 |
| CHB-MIT | AUC-PR | 0.6047 | [`48fea7d9a2325051cf61`](../datasets/chbmit.md) | 24339810 |

CHB-MIT 沿用该 PAC 运行的富集划分（测试集 17,386 窗口，276 个阳性）。ISRUC 按单个 30 秒 epoch 评估，共 8,700 个测试 epoch。提交前逐项核对了三份划分的类别计数及数据清单的创建时间。
