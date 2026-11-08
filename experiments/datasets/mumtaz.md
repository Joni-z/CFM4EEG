# mumtaz 实验记录

[返回总入口](../README.md)

共 52 条记录。[完整精度及来源 CSV](mumtaz.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（52 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `c824a7256d7cecae24e3` | Duplex-A128<br>mumtaz-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.965392 |  | 0.954860 | 0.796932 |  |
| `2b517049d730481ad649` | Duplex-A128<br>mumtaz-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.981046 |  | 0.977511 | 0.873477 |  |
| `509a603a9cd22eb5c533` | Duplex-A128<br>mumtaz-paclock_tune_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.983777 |  | 0.980413 | 0.820955 |  |
| `dfb5f48ab400680b4016` | Duplex-A128<br>mumtaz-paclock_tune_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.979305 |  | 0.973646 | 0.844719 |  |
| `68fc72d5f1cd063a7f18` | biot<br>mumtaz-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.999895 |  | 0.999877 | 0.890000 |  |
| `b71c6454282323d3d7a0` | biot<br>mumtaz-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.983650 |  | 0.986396 | 0.835000 |  |
| `519f29a71eb72cfa56c7` | biot<br>mumtaz-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.985252 |  | 0.982252 | 0.800000 |  |
| `3b5fac5afe77b2802246` | biot<br>mumtaz-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.957337 |  | 0.942186 | 0.797866 |  |
| `1aac231e28e6a5538015` | biot<br>mumtaz-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.972875 |  | 0.964317 | 0.797866 |  |
| `9bd877afe94cdc9bbba0` | biot<br>mumtaz-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.947783 |  | 0.929609 | 0.799289 |  |
| `4e7387c33451337d1a09` | cbramod<br>mumtaz-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.974136 |  | 0.968988 | 0.833333 |  |
| `02f997f9ba0f75bf0408` | cbramod<br>mumtaz-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.982953 |  | 0.980427 | 0.800000 |  |
| `02b8c9d4c5ceb45553bf` | cbramod<br>mumtaz-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.936348 |  | 0.933497 | 0.800000 |  |
| `ed29c2bd28b4b37ce087` | cbramod<br>mumtaz-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.904001 |  | 0.924008 | 0.801343 |  |
| `4f88d4f3c9bd58cef31c` | cbramod<br>mumtaz-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.881491 |  | 0.873284 | 0.811200 |  |
| `615ff6135fc5b7384f82` | cbramod<br>mumtaz-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.967597 |  | 0.964495 | 0.905653 |  |
| `a5665cfe291ddfe282b9` | cnn_transformer<br>mumtaz-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.877517 |  | 0.885043 | 0.823578 |  |
| `075e93cd44903c972006` | cnn_transformer<br>mumtaz-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.733780 |  | 0.805567 | 0.703333 |  |
| `6aef01b8ebe1982a4827` | cnn_transformer<br>mumtaz-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.937012 |  | 0.917492 | 0.797033 |  |
| `4aa325386401c3187d23` | contrawr<br>mumtaz-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.973535 |  | 0.965865 | 0.798455 |  |
| `9395936e3825772a0458` | contrawr<br>mumtaz-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.934023 |  | 0.920602 | 0.793455 |  |
| `341e6e3616b1ec4c03bb` | contrawr<br>mumtaz-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.970188 |  | 0.963888 | 0.724167 |  |
| `98a72f4d95738ad6161e` | eegpt<br>mumtaz-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.939029 |  | 0.937857 | 0.857613 |  |
| `b4c0fbbf3e86351c6332` | eegpt<br>mumtaz-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.990273 |  | 0.988874 | 0.945638 |  |
| `d71ddd4f61afbd3c4b40` | eegpt<br>mumtaz-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.956908 |  | 0.954801 | 0.851223 |  |
| `031f6e616e2359483921` | eegpt<br>mumtaz-eegpt_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.880933 |  | 0.893231 | 0.804496 |  |
| `e4d511260b59eb42fb7e` | eegpt<br>mumtaz-eegpt_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.644925 |  | 0.677183 | 0.500000 |  |
| `1d31a2a5e54023c92478` | eegpt<br>mumtaz-eegpt_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.826466 |  | 0.851261 | 0.781663 |  |
| `bc5511374c516c397ac3` | feat_best<br>mumtaz-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.932650 |  | 0.916330 | 0.800488 |  |
| `0ef29144604a5e4b09f1` | feat_best<br>mumtaz-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.932650 |  | 0.916330 | 0.800488 |  |
| `0fce82495eb167867e5a` | feat_best<br>mumtaz-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.932650 |  | 0.916330 | 0.800488 |  |
| `3c7a77d96d93d66c426e` | ffcl<br>mumtaz-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.966889 |  | 0.958129 | 0.762033 |  |
| `f9c65695f5dc5332e90d` | ffcl<br>mumtaz-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.966935 |  | 0.958895 | 0.806444 |  |
| `e0fade063d62cdc5b089` | ffcl<br>mumtaz-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.974164 |  | 0.968947 | 0.803333 |  |
| `5aad2a92ece6e26253b9` | labram<br>mumtaz-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.989055 |  | 0.985825 | 0.800955 |  |
| `d6ba23efd733e5bd8a60` | labram<br>mumtaz-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.929174 |  | 0.939388 | 0.800000 |  |
| `314a27ff632bda737f85` | labram<br>mumtaz-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.963333 |  | 0.965422 | 0.800000 |  |
| `4f5e4514548460cc09a0` | labram<br>mumtaz-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.987583 |  | 0.984758 | 0.803333 |  |
| `e61ac02fec67ff77c48c` | labram<br>mumtaz-labram_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.868305 |  | 0.880574 | 0.801667 |  |
| `168de3d62f7b5c026647` | labram<br>mumtaz-labram_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.813374 |  | 0.825474 | 0.799289 |  |
| `112cea61b244a3c247e7` | sparcnet<br>mumtaz-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.977189 |  | 0.975667 | 0.624167 |  |
| `a5e919375120a4541f46` | sparcnet<br>mumtaz-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.996312 |  | 0.994765 | 0.849218 |  |
| `5a0b590601b487b575b0` | sparcnet<br>mumtaz-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.999534 |  | 0.999429 | 0.864167 |  |
| `5a626b2a79ea3f9e73fc` | st_transformer<br>mumtaz-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.994149 |  | 0.993303 | 0.957707 |  |
| `017ec92bf13d2511620d` | st_transformer<br>mumtaz-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.998231 |  | 0.997670 | 0.830955 |  |
| `0ab3a17ba7e3f041ec94` | st_transformer<br>mumtaz-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.986512 |  | 0.982074 | 0.700000 |  |
| `383e236a9d7bd821ea65` | tfm<br>mumtaz-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.954277 |  | 0.949061 | 0.863110 |  |
| `e6275e6171d947b11ccd` | tfm<br>mumtaz-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.840645 |  | 0.870076 | 0.796078 |  |
| `f1e615d3e9efb286f46b` | tfm<br>mumtaz-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.827498 |  | 0.880303 | 0.840510 |  |
| `dfd1955456284a598810` | tfm<br>mumtaz-tfm_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.804601 |  | 0.831700 | 0.748122 |  |
| `aa44086c8ff6a1fdd969` | tfm<br>mumtaz-tfm_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.643221 |  | 0.664073 | 0.500000 |  |
| `d553ec840646c25a1635` | tfm<br>mumtaz-tfm_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.726167 |  | 0.736245 | 0.500000 |  |
