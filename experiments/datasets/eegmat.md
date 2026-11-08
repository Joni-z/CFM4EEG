# eegmat 实验记录

[返回总入口](../README.md)

共 16 条记录。[完整精度及来源 CSV](eegmat.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（16 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `5956ba34ba3742d477e8` | Duplex-A128<br>eegmat-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.469069 |  | 0.726261 | 0.558957 |  |
| `fbe25ab62ff3277fe7aa` | biot<br>eegmat-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.381837 |  | 0.660431 | 0.517290 |  |
| `6dbfd9f262702dd9203a` | biot<br>eegmat-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.336480 |  | 0.649093 | 0.571712 |  |
| `6ca1611fb9cf2436cfc6` | cbramod<br>eegmat-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.732289 |  | 0.855655 | 0.717829 |  |
| `330869e4412edbe2bd9e` | cbramod<br>eegmat-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.446564 |  | 0.707200 | 0.586451 |  |
| `75cbb3b1aa934b30d539` | cnn_transformer<br>eegmat-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.637600 |  | 0.804989 | 0.500000 |  |
| `fd7979a831c771b4c77a` | contrawr<br>eegmat-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.543776 |  | 0.772180 | 0.580499 |  |
| `5d7b3b186bcefba0c83b` | eegpt<br>eegmat-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.486377 |  | 0.681618 | 0.576814 |  |
| `821baa8d8721357adbb5` | eegpt<br>eegmat-eegpt_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.386201 |  | 0.690157 | 0.500000 |  |
| `b76589b1a498ee14fd83` | ffcl<br>eegmat-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.415588 |  | 0.715349 | 0.654337 |  |
| `f7d9f5f64ec87cf7f54f` | labram<br>eegmat-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.715239 |  | 0.821358 | 0.693736 |  |
| `ebf41398e2ca35893036` | labram<br>eegmat-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.344557 |  | 0.657951 | 0.497449 |  |
| `5d3ec535702aab963ba4` | sparcnet<br>eegmat-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.505334 |  | 0.769062 | 0.594813 |  |
| `046689f698803dc1a39e` | st_transformer<br>eegmat-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.289435 |  | 0.551587 | 0.500000 |  |
| `4e3f340be0f9a87eaeb7` | tfm<br>eegmat-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.381742 |  | 0.692744 | 0.509921 |  |
| `1471ac0b60f0724d0989` | tfm<br>eegmat-tfm_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.240152 |  | 0.463294 | 0.500000 |  |
