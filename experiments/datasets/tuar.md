# TUAR 实验记录

[返回总入口](../README.md)

共 93 条记录。[完整精度及来源 CSV](tuar.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（93 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `0a0608abffa898269be2` | CfFT其他前端<br>tuar-factorized_f0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.617019 |  |  |  | 0.755495 | 0.740975 |
| `d0162b15df4306706b9d` | CfFT其他前端<br>tuar-factorized_f1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.613581 |  |  |  | 0.752735 | 0.739112 |
| `8ff1552fb3a00cde4e68` | CfFT其他前端<br>tuar-factorized_f1_scale03_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.609112 |  |  |  | 0.749740 | 0.736170 |
| `a388f515136418b6739d` | CfFT其他前端<br>tuar-factorized_f1_scale03_mi2101x | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.631539 |  |  |  | 0.765290 | 0.751563 |
| `2049d73c70a4934dc142` | CfFT其他前端<br>tuar-factorized_f1_scale1_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.642577 |  |  |  | 0.773051 | 0.757087 |
| `610c5205c47d3069f9fc` | CfFT其他前端<br>tuar-factorized_f1_scale1_mi2101x | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.623079 |  |  |  | 0.757445 | 0.747172 |
| `30d91c989da8ca28ab90` | CfFT其他前端<br>tuar-factorized_f2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.686591 |  |  |  | 0.800670 | 0.787811 |
| `f85356126022c9cd81ee` | CfFT其他前端<br>tuar-factorized_f2_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.672296 |  |  |  | 0.790012 | 0.777823 |
| `9abc35eb1b63f70a6b60` | CfFT其他前端<br>tuar-factorized_f3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.726483 |  |  |  | 0.824512 | 0.816077 |
| `b452d108da94e7abceb7` | CfFT其他前端<br>tuar-factorized_f4_aug_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.719591 |  |  |  | 0.820102 | 0.810263 |
| `f322a9f13759f9c0f4d3` | CfFT其他前端<br>tuar-factorized_f4_joint_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.681251 |  |  |  | 0.796850 | 0.783195 |
| `df8b184d9d74c727ab1b` | Duplex<br>tuar-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.623728 |  |  |  | 0.761197 | 0.743587 |
| `8bc46230211f3499fd5e` | Duplex<br>tuar-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.626073 |  |  |  | 0.760891 | 0.748211 |
| `938890d32bfbcef580fe` | Duplex<br>tuar-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.656715 |  |  |  | 0.782054 | 0.766874 |
| `226bf23560d2fe6e1fb8` | Duplex-A128<br>tuar-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.592267 |  |  |  | 0.737944 | 0.725837 |
| `131bb4786cf857a1b13d` | Duplex-A128<br>tuar-paclock_ptF_t1 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.559587 |  |  |  | 0.715853 | 0.699594 |
| `8ed6cdfb4a35ade9df92` | Duplex-A128<br>tuar-paclock_ptF_t2 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.523300 |  |  |  | 0.692759 | 0.671667 |
| `ea4a72d56c9ae93800f4` | Duplex-A128<br>tuar-a128_r1_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.661478 |  |  |  | 0.781029 | 0.773906 |
| `3d18860ed0304654d0ca` | Duplex-A128<br>tuar-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.628927 |  |  |  | 0.764369 | 0.746187 |
| `1a1d7d88506e063ca7c0` | Duplex-A128<br>tuar-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.578968 |  |  |  | 0.737162 | 0.698715 |
| `3015be4e19e170a25926` | Duplex-A128<br>tuar-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.651844 |  |  |  | 0.777490 | 0.765232 |
| `9aa6a2715e5aa6512598` | Duplex-A128<br>tuar-paclock_tune_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.657561 |  |  |  | 0.780571 | 0.769974 |
| `9d870966f086631e4284` | Duplex-A128<br>tuar-paclock_tune_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.637982 |  |  |  | 0.770436 | 0.753231 |
| `7dd5ea7b112e7532096d` | Duplex-F192<br>tuar-cf2_v1d192_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.497808 |  |  |  | 0.678632 | 0.646517 |
| `f2193b2716e7c9f0cf94` | Duplex-F192<br>tuar-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.599711 |  |  |  | 0.742354 | 0.730730 |
| `f96a76d00cd44c91e5b3` | Duplex-F192<br>tuar-cf2_v1d192 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.577837 |  |  |  | 0.735788 | 0.703251 |
| `0960d3f5fed06c626517` | Duplex-F192<br>tuar-cf2_v1d192 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.596892 |  |  |  | 0.742504 | 0.726404 |
| `a00d6e523cc10d0ccc17` | Duplex-F192<br>tuar-cf2_v1d192_r1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.614548 |  |  |  | 0.751718 | 0.741367 |
| `6e46cbb97778f6b1f4cd` | Duplex-F192<br>tuar-cf2_v1d192_r2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.600000 |  |  |  | 0.742673 | 0.731456 |
| `101a1d7dc2b9ee2b0d65` | Duplex-F192<br>tuar-cf2_v1d192_r3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.610902 |  |  |  | 0.749613 | 0.738349 |
| `c8be31d10a176d663523` | Duplex-F192<br>tuar-cf2_v1d192_r4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.622394 |  |  |  | 0.757212 | 0.744134 |
| `ffe5ebf4798ff4d43e49` | Duplex-F192<br>tuar-cf2_v3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.545180 |  |  |  | 0.707616 | 0.696449 |
| `b9f7b74b9632d0b1276d` | Quad-Range<br>tuar-quad_range_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.713778 |  |  |  | 0.815776 | 0.806654 |
| `28f477a2d91be1ff3fe1` | Quad-Range<br>tuar-quad_range_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.724058 |  |  |  | 0.820932 | 0.814482 |
| `ca32c729c4202cd824d6` | Quad-Range<br>tuar-range_twoaxis_measured_20260917 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.704789 |  |  |  | 0.805506 | 0.801078 |
| `2f6953dd9f6b138c70da` | Quad-Range<br>tuar-range_twoaxis_measured_m200_20260917 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.688171 |  |  |  | 0.800221 | 0.787376 |
| `7a6209e10e54889987da` | Quad16<br>tuar-quad16_breadth_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.624324 |  |  |  | 0.761760 | 0.741883 |
| `9e1b1a19ceec6191da77` | Quad16<br>tuar-quad16_breadth_20260915 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.631739 |  |  |  | 0.762810 | 0.751351 |
| `ee95e69c92276e125604` | Quad16<br>tuar-quad16_breadth_20260915 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.651506 |  |  |  | 0.776090 | 0.766275 |
| `b00afb35aa02f09f9a9b` | biot<br>tuar-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.574976 |  |  |  | 0.728181 | 0.704698 |
| `bba5bda7407102a030d5` | biot<br>tuar-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.505323 |  |  |  | 0.687631 | 0.649747 |
| `fe5a7ff6764d11eeed80` | biot<br>tuar-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.638389 |  |  |  | 0.765889 | 0.757375 |
| `d1cb0ca071796a38db0a` | biot<br>tuar-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.498541 |  |  |  | 0.684890 | 0.627997 |
| `f505906d55e0bd3e5a8b` | biot<br>tuar-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.597483 |  |  |  | 0.741811 | 0.723491 |
| `fbca87a90ae1299bf392` | biot<br>tuar-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.583855 |  |  |  | 0.732447 | 0.715610 |
| `60281dd5fef3f0c4c6fd` | cbramod<br>tuar-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.714687 |  |  |  | 0.815321 | 0.807795 |
| `ba45bbfa0e300a6ca557` | cbramod<br>tuar-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.685855 |  |  |  | 0.797212 | 0.786862 |
| `a65fa853e095ae92f762` | cbramod<br>tuar-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.693152 |  |  |  | 0.801019 | 0.794356 |
| `17c0b03b93db4773aac6` | cbramod<br>tuar-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.650329 |  |  |  | 0.772852 | 0.761370 |
| `c75a9f4b672763b15595` | cbramod<br>tuar-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.670517 |  |  |  | 0.787140 | 0.776672 |
| `97b3ba480d56c3054f9e` | cbramod<br>tuar-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.639083 |  |  |  | 0.766677 | 0.754059 |
| `6ad34328b4b8b11794f5` | cbramod_paclockfe宿主转接<br>tuar-cbramod_add_cplpre | pretrained | 0 | completed_record<br>clinical_run | 0.587749 |  |  |  | 0.737420 | 0.711421 |
| `7f24bc809826a8449a16` | cbramod_paclockfe宿主转接<br>tuar-cbramod_add_rawpre | pretrained | 0 | completed_record<br>clinical_run | 0.681113 |  |  |  | 0.792666 | 0.785239 |
| `c4cecb3a4ba974974de2` | cnn_transformer<br>tuar-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.712021 |  |  |  | 0.813865 | 0.805902 |
| `a95c9d15cee770e6f8f5` | cnn_transformer<br>tuar-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.695851 |  |  |  | 0.800559 | 0.797666 |
| `56249c97c9c4732b442e` | cnn_transformer<br>tuar-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.686542 |  |  |  | 0.800237 | 0.784794 |
| `c88355f44c44895bffff` | contrawr<br>tuar-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.662997 |  |  |  | 0.784210 | 0.768564 |
| `944227288b128849cae4` | contrawr<br>tuar-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.695037 |  |  |  | 0.804902 | 0.793411 |
| `c894cc774339906435b1` | contrawr<br>tuar-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.721063 |  |  |  | 0.820036 | 0.812381 |
| `8d01101c79a2fe9c50d9` | eegpt<br>tuar-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.523458 |  |  |  | 0.698179 | 0.665186 |
| `28abb4c81d30b1fa7172` | eegpt<br>tuar-eegpt_scratch | scratch | 0 | completed_record<br>clinical_run | 0.581997 |  |  |  | 0.727900 | 0.719999 |
| `f033a5b305f222992e55` | eegpt<br>tuar-eegpt_scratch | scratch | 1 | completed_record<br>clinical_run | 0.543473 |  |  |  | 0.709060 | 0.687982 |
| `30f608d4663e091f5ff1` | eegpt<br>tuar-eegpt_scratch | scratch | 2 | completed_record<br>clinical_run | 0.545445 |  |  |  | 0.711587 | 0.687404 |
| `37937c887664b12322df` | ffcl<br>tuar-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.695940 |  |  |  | 0.799971 | 0.795781 |
| `47fbb47f8dc7b2856c0c` | ffcl<br>tuar-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.716513 |  |  |  | 0.812565 | 0.812283 |
| `2cd656b2a12891aadc6c` | ffcl<br>tuar-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.694922 |  |  |  | 0.803986 | 0.794430 |
| `0b14f30d95c73587120b` | labram<br>tuar-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.593125 |  |  |  | 0.743270 | 0.714984 |
| `587c7f9ff1949f590b77` | labram<br>tuar-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.614155 |  |  |  | 0.754110 | 0.734939 |
| `27e4ac7ebfe5f7def65b` | labram<br>tuar-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.656389 |  |  |  | 0.777922 | 0.770274 |
| `f53a1eea065a7b740410` | labram<br>tuar-labram_scratch | scratch | 0 | completed_record<br>clinical_run | 0.526535 |  |  |  | 0.699105 | 0.664067 |
| `0ee0cc59ec5dfa6cd086` | labram<br>tuar-labram_scratch | scratch | 1 | completed_record<br>clinical_run | 0.557544 |  |  |  | 0.713924 | 0.698902 |
| `7ae1d2d7723dd9954f9d` | labram<br>tuar-labram_scratch | scratch | 2 | completed_record<br>clinical_run | 0.541101 |  |  |  | 0.703439 | 0.688532 |
| `332b86df621d4faabaa6` | sparcnet<br>tuar-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.600978 |  |  |  | 0.748354 | 0.716177 |
| `b968a62e6557f87a4b48` | sparcnet<br>tuar-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.647319 |  |  |  | 0.772665 | 0.758321 |
| `7bd2201311dd1fe64f5a` | sparcnet<br>tuar-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.596521 |  |  |  | 0.742918 | 0.716770 |
| `8847144024ec65a826a5` | st_transformer<br>tuar-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.521289 |  |  |  | 0.682472 | 0.672493 |
| `724da660869c1b48b184` | st_transformer<br>tuar-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.518594 |  |  |  | 0.682911 | 0.670758 |
| `a1d05a197e3967aa92a1` | st_transformer<br>tuar-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.560645 |  |  |  | 0.703484 | 0.702201 |
| `47e1200773eb07701169` | tfm<br>tuar-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.647980 |  |  |  | 0.775583 | 0.758394 |
| `993ee2ad42737d4d0d8b` | tfm<br>tuar-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.641479 |  |  |  | 0.763766 | 0.757898 |
| `1e10cc271aee62761684` | tfm<br>tuar-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.646832 |  |  |  | 0.768326 | 0.761694 |
| `1dfe98586a7bbbe307bf` | tfm<br>tuar-tfm_scratch | scratch | 0 | completed_record<br>clinical_run | 0.364372 |  |  |  | 0.585343 | 0.557451 |
| `4412da325b8e60125382` | tfm<br>tuar-tfm_scratch | scratch | 1 | completed_record<br>clinical_run | 0.403618 |  |  |  | 0.611382 | 0.587647 |
| `2d0c29a3ca4014f7c9ee` | tfm<br>tuar-tfm_scratch | scratch | 2 | completed_record<br>clinical_run | 0.363011 |  |  |  | 0.585390 | 0.552835 |
| `4a8d7928940b9453d7df` | 频带关系内容版<br>tuar-paclock_pac | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.577960 |  |  |  | 0.730511 | 0.711920 |
| `95f6c458bf5d9823e1cf` | 频带关系内容版<br>tuar-paclock_rot | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.556807 |  |  |  | 0.716920 | 0.696149 |
| `2e4dc687977b34a497c7` | 频带关系内容版<br>tuar-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.556807 |  |  |  | 0.716920 | 0.696149 |
| `2663b205a1b942ac8c08` | 频带关系内容版<br>tuar-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.597054 |  |  |  | 0.746183 | 0.720262 |
| `cebc5fd16072ffc57227` | 频带关系内容版<br>tuar-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.597533 |  |  |  | 0.746006 | 0.723004 |
| `0767d5f978eb546c4f2f` | 频带波形原始版<br>tuar-paclock_raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.628909 |  |  |  | 0.761251 | 0.749097 |
| `182f7672e8ef9b02cd25` | 频带波形原始版<br>tuar-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.628909 |  |  |  | 0.761251 | 0.749097 |
| `b076cf260c4cb90b49df` | 频带波形原始版<br>tuar-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.614243 |  |  |  | 0.755337 | 0.732816 |
| `be5be4343d0bb665e5fa` | 频带波形原始版<br>tuar-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.627391 |  |  |  | 0.760769 | 0.746905 |
