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

## 四个数据集的配对训练（2026-11-24）

四项 AAC 使用 seed 0，各自沿用已完成的双轴 PAC 运行的训练配置，将 `coupling_measure` 设为 `aac`；配置与对应 PAC 运行只差 `coupling_measure`、`name`、`group` 和 `save_checkpoint`。PAC 不重跑，直接用已有的 seed 0 记录。

| 数据集 | 主指标 | PAC 对照 | PAC 记录 | Torch AAC job |
|---|---|---:|---|---:|
| TUSZ | AUC-PR | 0.6581 | [`3a0d527cd2686f53f7ea`](../datasets/tusz.md) | 24339795 |
| IIIC | κ | 0.4665 | [`5debb59344678e8d4308`](../datasets/iiic.md) | 24339803 |
| ISRUC | κ | 0.7680 | [`bb3b9758768984586dd5`](../datasets/isruc.md) | 24339807 |
| CHB-MIT | AUC-PR | 0.6047 | [`48fea7d9a2325051cf61`](../datasets/chbmit.md) | 24339810 |

CHB-MIT 沿用该 PAC 运行的富集划分（测试集 17,386 窗口，276 个阳性）。ISRUC 按单个 30 秒 epoch 评估，共 8,700 个测试 epoch。提交前逐项核对了三份划分的类别计数及数据清单的创建时间。

### 结果：PAC、AAC 与最强基线

| 数据集 | 主指标 | PAC | AAC | PAC − AAC | 最强基线（3 seed 均值） | PAC − 基线 | AAC − 基线 |
|---|---|---:|---:|---:|---|---:|---:|
| TUEV | κ | 0.6736 | 0.6421 | +0.0315 | REVE 预训练 0.6846 | −0.0110 | −0.0425 |
| TUSZ | AUC-PR | 0.6581 | 0.6427 | +0.0154 | FFCL 0.5449 | +0.1132 | +0.0978 |
| IIIC | κ | 0.4665 | **0.4852** | −0.0187 | REVE 预训练 0.4363 | +0.0302 | +0.0489 |
| ISRUC | κ | 0.7680 | 0.7446 | +0.0234 | CBraMod 预训练 0.7540 | +0.0140 | −0.0094 |
| CHB-MIT | AUC-PR | 0.6047 | 0.5768 | +0.0279 | TFM 预训练 0.6269 | −0.0222 | −0.0501 |

TUEV 一行的 PAC 和 AAC 都是 Torch 上的 seed 2，与上文相同；TUEV 主模型 checkpoint（amd，0.7134）比 REVE 高 0.0288。其余四行的 PAC 来自已有记录（TUSZ、IIIC、ISRUC 在 amd 上训练，CHB-MIT 在 145 上训练），AAC 在 Torch A100 上训练。

| AAC 运行 | balanced acc（PAC / AAC） | 最佳验证分（PAC / AAC） | 训练 epoch（PAC / AAC） |
|---|---|---|---|
| TUSZ | 0.6676 / 0.5838 | 0.3443 / 0.3718 | 2 / 3，均为 patience 停止 |
| IIIC | 0.5522 / 0.5706 | 0.5388 / 0.5472 | 20 / 20 |
| ISRUC | 0.7921 / 0.7774 | 0.7488 / 0.7444 | 20 / 20 |
| CHB-MIT | 0.5524 / 0.5289 | 0.5644 / 0.3071 | 5 / 2，均为 patience 停止 |

最强基线的选法与 [baseline-results.json](../../papers/demodulation/evidence/baseline-results.json) 相同：只用正式配置（不含 archive 和 rebuild 运行），计入全部已记录的 seed，测试窗口与 PAC 记录一致，至少三个 seed，取均值最高者。TUEV 和 CHB-MIT 的结果与该文件一致。TUSZ 和 CHB-MIT 的基线与 CfFT 用同一训练框架，主指标都是梯形 PR-AUC。

### 说明

- 单 seed 对单 seed；基线是三个 seed 的均值。TUEV 上这个配置在 amd 上三个 seed 的 κ 标准差是 0.014，与表中多数差值同一量级。
- PAC 与 AAC 不在同一台机器上训练。TUEV 上同一配置换到 Torch 重训，κ 比 amd 低 0.04，所以这四行的 PAC − AAC 包含机器差异。
- 这四条 PAC 记录没有保存逐窗口预测，未做配对 bootstrap。
