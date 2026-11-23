# TUEV 主模型 checkpoint

2026-11-23 起，CfFT 主模型（两轴编码器、measured PAC token）在 TUEV 上的结果统一用下面这个 checkpoint 报告。

| 项目 | 内容 |
|---|---|
| 运行 | `tuev-range_twoaxis_measured_20260917`，seed 2，2026-09 在 amd（AMD Instinct MI210）训练 |
| 台账记录 | `55ab79f81fa6a6377b3e`（[TUEV 页面](../datasets/tuev.md)）· model_version `cd8ca1e167e682dd` · recipe `d803dadf8dc80958` |
| 配置 | 模型和训练设置与 [tuev-scratch.yaml](../../configs/model/tuev-scratch.yaml) 相同，seed 2；1,224,486 参数 |
| 选中 epoch | 5（验证集 κ 0.6387） |
| 测试集（shared16，29,421 窗口） | **κ 0.7134** · balanced acc 0.6404 · weighted F1 0.8513 · accuracy 0.8534 |
| checkpoint SHA-256 | `1fb2b132f935d890373f38f47ee2b5e99c796b3d5d9b76773c74b94b3e8b1a3c` |
| 存放位置 | Hugging Face `JoniZz/CFM4EEG-checkpoint-archive`：`objects/1f/1fb2b132….bin`；Mac mini：`.local/legacy/checkpoints/objects/1fb2b132….pt` |

## 同配置的三个 seed

| seed | 验证集 κ | 测试集 κ | balanced acc | 台账记录 |
|---|---:|---:|---:|---|
| 0 | 0.6260 | 0.6846 | 0.6366 | `491c3b552f8c14fbfcad` |
| 1 | 0.5971 | 0.6985 | 0.5782 | `930fc363622fe0f4f996` |
| **2** | **0.6387** | **0.7134** | **0.6404** | `55ab79f81fa6a6377b3e` |

三个 seed 的测试集 κ 为 0.6988 ± 0.0144（均值 ± 样本标准差）。seed 2 的验证集 κ 也是三个 seed 中最高的，所以这个选择与“按验证集 κ 选 seed”的规则一致。需要报告 seed 方差时用这里的均值和标准差。

## 复现检查

**重新评估。** 2026-11-23 在 Torch A100 上用当前代码重新评估三个 checkpoint（job 24157292，严格加载 state dict，数据 manifest `5fd1f6a4…`）。三个测试集 κ 与当时记录的差都是 0，验证集指标也一致。数据、预处理、推理和评估代码自 9 月以来都没有变化。

**训练代码。** amd 训练代码快照保存在 `.local/legacy/code/twoaxis-runtime`，2026-09-19 抓取，包含当时未提交的改动。在 Torch CPU 上，分别用快照代码和当前代码、相同 seed 构建模型，并在固定 batch 上做一次训练模式的前向和反向：初始化、训练输出（含数据增强）、loss、梯度、eval 输出和 Python 随机数状态全部逐位相同。

## 和新训练结果比较时

在 Torch A100 上用同一配置重训 seed 2，测试集 κ 为 0.6736（[AAC 对照](cfft-aac-control-20261123.md)中的 PAC 臂）。上面已经确认代码和数据相同，所以这个差别来自不同 GPU 上的训练过程。在 Torch 上重训的新方法或对照，应当和 Torch 上同配置的 PAC 重训比较，不和 0.7134 比较。

[完整数值与哈希](tuev-main-checkpoint-20261123.json)
