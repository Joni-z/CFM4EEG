# ADFD 实验记录

[返回总入口](../README.md)

共 111 条记录。[完整精度及来源 CSV](adfd.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## Accepted Medformer derivative19x2000float32microvolts/100. Storagebridge returnsphysicaluV; originalengine divides100.19distinct random trainable ordi（1 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `611acc095e10d43483c7` | labram<br>adfd-seed0 | scratch | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.218507 |

## protocol_not_bound（110 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `696b6d3ec04d855a0568` | Duplex<br>adfd-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.315558 |  |  |  | 0.498193 | 0.526981 |
| `eefe50b98a03bdb7a69e` | Duplex<br>adfd-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.260409 |  |  |  | 0.466668 | 0.491128 |
| `e703c23251def22fdc3b` | Duplex<br>adfd-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.240906 |  |  |  | 0.454013 | 0.478644 |
| `ab99b93a290612f04b5c` | Duplex-A128<br>adfd-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.124810 |  |  |  | 0.378556 | 0.428133 |
| `07bef2dd3c80fd095a82` | Duplex-A128<br>adfd-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.200710 |  |  |  | 0.452653 | 0.481738 |
| `51e173ccb023ff6aedab` | Duplex-A128<br>adfd-paclock_ptF_lr10th | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.194850 |  |  |  | 0.424122 | 0.457219 |
| `98b39899235acc5d4ada` | Duplex-A128<br>adfd-paclock_ptF_lr3rd | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.278882 |  |  |  | 0.482472 | 0.513495 |
| `c18a99cb89beb03ceff3` | Duplex-A128<br>adfd-a128_r1_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.396137 |  |  |  | 0.560447 | 0.583207 |
| `6a18a877a66f1e49cf96` | Duplex-A128<br>adfd-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.361278 |  |  |  | 0.561695 | 0.581352 |
| `924dc09762ed712e3d19` | Duplex-A128<br>adfd-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.341417 |  |  |  | 0.511936 | 0.535840 |
| `b9f4c316fe685bebe68d` | Duplex-A128<br>adfd-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.232053 |  |  |  | 0.440469 | 0.486864 |
| `d20be60c24384522037d` | Duplex-F192<br>adfd-cf2_v1d192_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.250799 |  |  |  | 0.466110 | 0.469139 |
| `9c9f7e4b0abe037b83d8` | Duplex-F192<br>adfd-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.215094 |  |  |  | 0.460816 | 0.486938 |
| `0613972b61e4c1a3378c` | Duplex-F192<br>adfd-cf2_v1d192 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.086902 |  |  |  | 0.381438 | 0.408159 |
| `9c7f9cbdcffa38bac8d8` | Duplex-F192<br>adfd-cf2_v1d192 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.319539 |  |  |  | 0.526906 | 0.555354 |
| `c054e5227d555eb56375` | Duplex-F192<br>adfd-cf2_v1d192_r1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.173806 |  |  |  | 0.424081 | 0.459760 |
| `7015acf67f378076d318` | Duplex-F192<br>adfd-cf2_v1d192_r2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.248983 |  |  |  | 0.466442 | 0.503039 |
| `22cba970b62848aecb9b` | Duplex-F192<br>adfd-cf2_v1d192_r3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.163068 |  |  |  | 0.430065 | 0.456621 |
| `f0b7a0e91346dde1cedb` | Duplex-F192<br>adfd-cf2_v1d192_r4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.187604 |  |  |  | 0.429330 | 0.462218 |
| `5f9efaf91e6c9203f0cc` | Duplex-F192<br>adfd-cf2_v3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.079526 |  |  |  | 0.360998 | 0.399565 |
| `623d4fcf6de34a3c2ae8` | Quad-Range<br>adfd-quad_range_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.123139 |  |  |  | 0.377586 | 0.431881 |
| `e1c14cc79251e15f32f1` | Quad-Range<br>adfd-quad_range_20260916 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.235864 |  |  |  | 0.459616 | 0.494436 |
| `00ad695db26b261ac34e` | Quad-Range<br>adfd-range_twoaxis_measured_20260917 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.271675 |  |  |  | 0.476359 | 0.507488 |
| `eb7ac785e04bd4c9f4d6` | Quad-Range<br>adfd-range_twoaxis_measured_m200_20260917 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.271659 |  |  |  | 0.478678 | 0.496191 |
| `58e2bcc6b9718ad6e0af` | Quad16<br>adfd-quad16_breadth_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.161760 |  |  |  | 0.421411 | 0.442842 |
| `46dc2f8cd877d91992f5` | Quad16<br>adfd-quad16_breadth_20260915 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.277271 |  |  |  | 0.485117 | 0.527340 |
| `ebc6cc37361f266e789b` | Quad16<br>adfd-quad16_breadth_20260915 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.251125 |  |  |  | 0.461488 | 0.502101 |
| `dfc63deae6a0dfebc0d5` | biot<br>adfd-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.325437 |  |  |  | 0.510473 | 0.537922 |
| `77fd801bcddb7e1fa0a3` | biot<br>adfd-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.252099 |  |  |  | 0.467054 | 0.483591 |
| `6364fb50e869ae4ace43` | biot<br>adfd-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.251601 |  |  |  | 0.471049 | 0.499431 |
| `5aa9b954624d54e0f2b9` | biot<br>adfd-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.325544 |  |  |  | 0.527943 | 0.554282 |
| `72c4c14b8a1f669f9a07` | biot<br>adfd-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.343563 |  |  |  | 0.544935 | 0.571581 |
| `1b72572e44e7a6b90e71` | biot<br>adfd-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.263260 |  |  |  | 0.503196 | 0.501564 |
| `7143c18532e46e6fbde0` | cbramod<br>adfd-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.226858 |  |  |  | 0.459932 | 0.490349 |
| `c51337b63597229c020d` | cbramod<br>adfd-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.171978 |  |  |  | 0.434520 | 0.459801 |
| `b3db022355f67555c442` | cbramod<br>adfd-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.155339 |  |  |  | 0.428957 | 0.448885 |
| `babc3ebb9d625d8d6305` | cbramod<br>adfd-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.256580 |  |  |  | 0.469399 | 0.498557 |
| `e5068d75dccba44b1b36` | cbramod<br>adfd-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.255115 |  |  |  | 0.471128 | 0.505911 |
| `46a701d68694caee1235` | cbramod<br>adfd-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.309168 |  |  |  | 0.502405 | 0.538450 |
| `c88e851dddf82b0bf6b7` | cbramod_paclockfe宿主转接<br>adfd-cbramod_crofremo | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.251767 |  |  |  | 0.469274 | 0.503060 |
| `8b2e2d3298f69a06b18b` | cbramod_paclockfe宿主转接<br>adfd-cbramod_crofremo | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.235974 |  |  |  | 0.467522 | 0.499384 |
| `7dcc1d78d092a05fb85f` | cbramod_paclockfe宿主转接<br>adfd-cbramod_crofremo | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.218643 |  |  |  | 0.451561 | 0.493947 |
| `9bee46f26c978379bb07` | cnn_transformer<br>adfd-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.224566 |  |  |  | 0.461745 | 0.493210 |
| `c45bb3574c7663811122` | cnn_transformer<br>adfd-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.300050 |  |  |  | 0.492533 | 0.493982 |
| `4f29b1919ca4886edc5d` | cnn_transformer<br>adfd-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.251685 |  |  |  | 0.486950 | 0.514020 |
| `e1e96c3501c835ab5a52` | contrawr<br>adfd-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.140151 |  |  |  | 0.425348 | 0.408991 |
| `c308e407277860f8344d` | contrawr<br>adfd-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.218784 |  |  |  | 0.472338 | 0.489524 |
| `96d9dcd76a9e4e3e54b8` | contrawr<br>adfd-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.178386 |  |  |  | 0.450703 | 0.449682 |
| `a43c02f39e6f273e33b1` | csbrain<br>adfd-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.172342 |  |  |  | 0.425216 | 0.464015 |
| `f46ef808122590fe5049` | csbrain<br>adfd-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.220537 |  |  |  | 0.457631 | 0.494405 |
| `0e8dba01ecab89ac21a6` | csbrain<br>adfd-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.207474 |  |  |  | 0.444827 | 0.483806 |
| `6a6ba7aea20a83485087` | eegconformer<br>adfd-eegconformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.061998 |  |  |  | 0.339848 | 0.393082 |
| `644f7746adb646ffdba1` | eegconformer<br>adfd-eegconformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.122578 |  |  |  | 0.388605 | 0.431429 |
| `e43f56444caa108f936a` | eegconformer<br>adfd-eegconformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.278879 |  |  |  | 0.488613 | 0.504432 |
| `ecc9ac083d720c66c4c8` | eegconformer<br>adfd-eegconformer_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.065389 |  |  |  | 0.347679 | 0.395594 |
| `38215866d3e3015bb1c6` | eegconformer<br>adfd-eegconformer_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.291679 |  |  |  | 0.484474 | 0.497740 |
| `f058108ff1c7d2df066c` | eegconformer<br>adfd-eegconformer_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.061998 |  |  |  | 0.339848 | 0.393082 |
| `e2f53e6376c9ad2e4f2f` | eegconformer<br>adfd-eegconformer_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.285649 |  |  |  | 0.485092 | 0.513188 |
| `f6454f7af8d75ef4e39b` | eegnet<br>adfd-eegnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | -0.031192 |  |  |  | 0.308875 | 0.179323 |
| `58404a0fe9d0cb17a3e9` | eegnet<br>adfd-eegnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.006446 |  |  |  | 0.340613 | 0.253723 |
| `c53912d804aca96ef7d6` | eegnet<br>adfd-eegnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | -0.010117 |  |  |  | 0.327642 | 0.277004 |
| `0f8272f429f2031dc2cd` | eegnet<br>adfd-eegnet_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | -0.045160 |  |  |  | 0.298022 | 0.215827 |
| `8de060aac126f13b9f74` | eegnet<br>adfd-eegnet_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | -0.038338 |  |  |  | 0.301953 | 0.158185 |
| `ae408c58d50aeee54c50` | eegnet<br>adfd-eegnet_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.005182 |  |  |  | 0.337317 | 0.224322 |
| `d912c0532e9c24cb2264` | eegnet<br>adfd-eegnet_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | -0.031192 |  |  |  | 0.308875 | 0.179323 |
| `8a65957188ba249d8fff` | eegpt<br>adfd-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.324726 |  |  |  | 0.500753 | 0.530828 |
| `f1e5acfb1250314a0473` | eegpt<br>adfd-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.234662 |  |  |  | 0.474339 | 0.489960 |
| `5751a49a9fed998f0476` | eegpt<br>adfd-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.334729 |  |  |  | 0.514250 | 0.544072 |
| `066b8205013edf6f5785` | eegpt<br>adfd-eegpt_scratch | scratch | 0 | completed_record<br>clinical_run | 0.106670 |  |  |  | 0.389707 | 0.389091 |
| `c00efd8a4ed54a5908a5` | eegpt<br>adfd-eegpt_scratch | scratch | 1 | completed_record<br>clinical_run | 0.286605 |  |  |  | 0.476164 | 0.515987 |
| `67b8e219a706cbae8b84` | eegpt<br>adfd-eegpt_scratch | scratch | 2 | completed_record<br>clinical_run | 0.274061 |  |  |  | 0.495931 | 0.530383 |
| `a946862cb7befda1f1a4` | feat_best<br>adfd-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.129978 |  |  |  | 0.400714 | 0.429629 |
| `feb5dfe08dd4ab2a8e04` | feat_best<br>adfd-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.129978 |  |  |  | 0.400714 | 0.429629 |
| `35b52b7eb860fb5c527e` | feat_best<br>adfd-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.129978 |  |  |  | 0.400714 | 0.429629 |
| `5bfcd804ebda19625aa0` | ffcl<br>adfd-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.101636 |  |  |  | 0.391545 | 0.363099 |
| `7cb189f46a905b5c5a3e` | ffcl<br>adfd-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.036145 |  |  |  | 0.344946 | 0.296045 |
| `41b8fc0896160d1595b3` | ffcl<br>adfd-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.124610 |  |  |  | 0.408843 | 0.389800 |
| `14bebc2a1dd94e713d10` | labram<br>adfd-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.218507 |
| `92797966883f62813704` | labram<br>adfd-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.202336 |  |  |  | 0.441333 | 0.473874 |
| `792838ad68f8e6b400d8` | labram<br>adfd-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.168383 |  |  |  | 0.415409 | 0.462296 |
| `f936a3550edaa6332eef` | labram<br>adfd-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.218507 |
| `d18e11f684f86d5ebd85` | labram<br>adfd-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.218507 |
| `ea0b164e5417199cef36` | labram<br>adfd-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.128492 |  |  |  | 0.394769 | 0.435187 |
| `a00e31e810afd269e021` | labram<br>adfd-labram_pt_lr1e4 | pretrained | 0 | completed_record<br>clinical_run | 0.202336 |  |  |  | 0.441333 | 0.473874 |
| `5bcc8a8b64b27383ae87` | labram<br>adfd-labram_pt_lr2e4 | pretrained | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.218507 |
| `1354912197d8d0f2a4a1` | labram<br>adfd-labram_scratch | scratch | 0 | completed_record<br>clinical_run | 0.268740 |  |  |  | 0.484026 | 0.516875 |
| `7cebfe57cf06dfee0f2b` | labram<br>adfd-labram_scratch | scratch | 1 | completed_record<br>clinical_run | 0.315765 |  |  |  | 0.521585 | 0.522137 |
| `9ac0c3b095e6d9a1edf5` | labram<br>adfd-labram_scratch | scratch | 2 | completed_record<br>clinical_run | 0.301364 |  |  |  | 0.491960 | 0.507175 |
| `f1b161ea5bec5f6b6441` | reve<br>adfd-reve_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.280559 |  |  |  | 0.483558 | 0.510556 |
| `1fbd016e9fca730da56f` | reve<br>adfd-reve_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.247705 |  |  |  | 0.462543 | 0.488016 |
| `db8773e6e8b220943842` | reve<br>adfd-reve_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.362617 |  |  |  | 0.555566 | 0.580271 |
| `769da4a1f326ca3097b0` | reve<br>adfd-reve_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run | 0.009855 |  |  |  | 0.343806 | 0.260744 |
| `8bb5e949c102c8cdeac7` | sparcnet<br>adfd-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.120605 |  |  |  | 0.389458 | 0.422419 |
| `61f80f66ebeef63dd555` | sparcnet<br>adfd-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.101628 |  |  |  | 0.385604 | 0.366030 |
| `8ccf8fb969873319f4a8` | sparcnet<br>adfd-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.061110 |  |  |  | 0.374727 | 0.334660 |
| `afb2aeb5196028a47a58` | st_transformer<br>adfd-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.252463 |  |  |  | 0.467646 | 0.473086 |
| `05681e8a299ec1381270` | st_transformer<br>adfd-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.389255 |  |  |  | 0.548779 | 0.570726 |
| `65ef73f254146092dd8e` | st_transformer<br>adfd-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.230084 |  |  |  | 0.454676 | 0.460124 |
| `57ffc47375b25af741be` | tfm<br>adfd-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.220020 |  |  |  | 0.464496 | 0.491844 |
| `69c7ce571e97fa0e5357` | tfm<br>adfd-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.167690 |  |  |  | 0.425215 | 0.455996 |
| `a89e7d5f44e813acec9f` | tfm<br>adfd-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.199970 |  |  |  | 0.449544 | 0.459665 |
| `690e452b728fe5ca9061` | tfm<br>adfd-tfm_scratch | scratch | 0 | completed_record<br>clinical_run | 0.230080 |  |  |  | 0.459850 | 0.485480 |
| `49060074ee5f862d7dd3` | tfm<br>adfd-tfm_scratch | scratch | 1 | completed_record<br>clinical_run | 0.182681 |  |  |  | 0.430191 | 0.433920 |
| `085224f43689853826ad` | tfm<br>adfd-tfm_scratch | scratch | 2 | completed_record<br>clinical_run | 0.256300 |  |  |  | 0.473155 | 0.489719 |
| `5b54d896995ef75647df` | 频带关系内容版<br>adfd-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.298640 |  |  |  | 0.505223 | 0.535491 |
| `409e8e49404addeefd0f` | 频带关系内容版<br>adfd-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.354062 |  |  |  | 0.522229 | 0.540349 |
| `753c013343e37d4e06e3` | 频带关系内容版<br>adfd-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.183673 |  |  |  | 0.423715 | 0.447362 |
| `2d2fdd1d0f8796a3865f` | 频带波形原始版<br>adfd-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.313630 |  |  |  | 0.497047 | 0.528920 |
| `d3affde5c6c16410804a` | 频带波形原始版<br>adfd-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.221696 |  |  |  | 0.455143 | 0.492010 |
| `fbbd8b50215db5911927` | 频带波形原始版<br>adfd-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.380058 |  |  |  | 0.552241 | 0.581678 |
