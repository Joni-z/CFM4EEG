# TUAB 实验记录

[返回总入口](../README.md)

共 123 条记录。[完整精度及来源 CSV](tuab.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（123 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `c643e8f72a38c08eeeee` | Duplex<br>tuab-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.883148 |  | 0.878355 | 0.795262 |  |
| `033d8b4499a27e956b0d` | Duplex<br>tuab-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.897466 |  | 0.888069 | 0.812344 |  |
| `7249da378e6a94444e4b` | Duplex<br>tuab-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.887193 |  | 0.878150 | 0.792594 |  |
| `2f012445b1d01e85df91` | Duplex-A128<br>tuab-paclock_duplex_pt | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.892318 |  | 0.886727 | 0.814047 |  |
| `5302b32a6557502d5ea4` | Duplex-A128<br>tuab-paclock_duplex_pt2 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.898943 |  | 0.893761 | 0.814967 |  |
| `19d4cadd2ebe766a8859` | Duplex-A128<br>tuab-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.886165 |  | 0.879887 | 0.810083 |  |
| `21edfc8cc4261c4212df` | Duplex-A128<br>tuab-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.893831 |  | 0.889164 | 0.805385 |  |
| `c2401ea3fb056a72ff87` | Duplex-A128<br>tuab-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.892259 |  | 0.882873 | 0.815690 |  |
| `88abdfe76f8a9eaa2e7b` | Duplex-A128<br>tuab-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.903728 |  | 0.894525 | 0.805965 |  |
| `024fc19ea5f75233ec0e` | Duplex-A128<br>tuab-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.897746 |  | 0.891299 | 0.817929 |  |
| `f9a85c1ad78f21592c31` | Duplex-F192<br>tuab-cf2_v1d192_ptR | checkpoint_loaded | 0 | budget_truncated<br>clinical_run |  | 0.887655 |  | 0.887657 | 0.813185 |  |
| `4de79bda5dd1681a0c2f` | Duplex-F192<br>tuab-cf2_v1d192 | not_explicitly_recorded | 0 | budget_truncated<br>clinical_run |  | 0.889547 |  | 0.883920 | 0.819616 |  |
| `5e943785fc1a8d10d4cb` | Duplex-F192<br>tuab-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.889547 |  | 0.883920 | 0.819616 |  |
| `4e2d61eb6ccd0e5a42cb` | Duplex-F192<br>tuab-cf2_v1d192 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.887447 |  | 0.876272 | 0.804459 |  |
| `890516d98b8aa393b45d` | Duplex-F192<br>tuab-cf2_v1d192 | not_explicitly_recorded | 1 | budget_truncated<br>clinical_run |  | 0.887447 |  | 0.876272 | 0.804459 |  |
| `223dc226386e19c939e7` | Duplex-F192<br>tuab-cf2_v1d192 | not_explicitly_recorded | 2 | budget_truncated<br>clinical_run |  | 0.904729 |  | 0.898680 | 0.829329 |  |
| `fc6e7470eabf9c8ecb62` | Duplex-F192<br>tuab-cf2_v3 | not_explicitly_recorded | 0 | budget_truncated<br>clinical_run |  | 0.889619 |  | 0.878811 | 0.807475 |  |
| `940d4d1fd52af260b58b` | Quad-Range<br>tuab-range_twoaxis_measured_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.891103 |  | 0.884346 | 0.803912 |  |
| `8ed6b5949bb08a43dfcb` | biot<br>tuab-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.865603 |  | 0.874961 | 0.797088 |  |
| `5c145c81f7ba93c89426` | biot<br>tuab-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.869304 |  | 0.869235 | 0.796647 |  |
| `464153fce83a448e070a` | biot<br>tuab-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.876155 |  | 0.874719 | 0.791334 |  |
| `6ceb416313d39a1e37aa` | biot<br>tuab-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.865475 |  | 0.865752 | 0.790293 |  |
| `eebd9f676363f265fd45` | biot<br>tuab-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.871429 |  | 0.876050 | 0.795518 |  |
| `04f349d3a8fbf3341234` | biot<br>tuab-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.867947 |  | 0.866498 | 0.786891 |  |
| `1f854ce83248236650a5` | biot<br>tuab-biot_scratch16_rebuild20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.886601 | 0.886604 | 0.876713 | 0.806863 |  |
| `64884208751e7ef8ab2e` | cbramod<br>tuab-cbramod_pac | pretrained | 0 | completed_record<br>clinical_run |  | 0.863879 |  | 0.867783 | 0.789257 |  |
| `d87dfae80baf2b81cb66` | cbramod<br>tuab-cbramod_pac | pretrained | 1 | completed_record<br>clinical_run |  | 0.861560 |  | 0.865560 | 0.787960 |  |
| `28fa00170a685fa5885e` | cbramod<br>tuab-cbramod_pac | pretrained | 2 | completed_record<br>clinical_run |  | 0.866130 |  | 0.863146 | 0.787212 |  |
| `a0a58316a204b20fd42b` | cbramod<br>tuab-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.865356 |  | 0.858635 | 0.789114 |  |
| `a66e8dcd40c9abf75a0a` | cbramod<br>tuab-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.881087 |  | 0.876532 | 0.796232 |  |
| `a9375b4c54cc68caf73c` | cbramod<br>tuab-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.892028 |  | 0.889991 | 0.810389 |  |
| `2957d14de694b432f3d9` | cbramod<br>tuab-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.876371 |  | 0.869741 | 0.797103 |  |
| `46355c35c4ea51ac713c` | cbramod<br>tuab-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.864285 |  | 0.853931 | 0.775298 |  |
| `746b9bd11a0b03a8a827` | cbramod<br>tuab-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.860725 |  | 0.858408 | 0.773265 |  |
| `7759f943422fd1e8de1d` | cbramod<br>tuab-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.876265 |  | 0.870251 | 0.800487 |  |
| `adc2eb700cc978966f14` | cbramod<br>tuab-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.872578 |  | 0.874417 | 0.798468 |  |
| `df5e68910e6baa11def0` | cbramod<br>tuab-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.888471 |  | 0.885767 | 0.801781 |  |
| `1ec6651011eaa8d0ff09` | cbramod<br>tuab-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.879337 |  | 0.872687 | 0.788057 |  |
| `7ef2970dd9db67a51337` | cbramod<br>tuab-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.869589 |  | 0.873003 | 0.798647 |  |
| `81a22e7ed4b37e09d555` | cbramod<br>tuab-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.894118 |  | 0.885771 | 0.802826 |  |
| `d6815858607ec8f0012d` | cbramod<br>tuab-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.894118 |  | 0.885771 | 0.802826 |  |
| `2f83455642eb338c12eb` | cbramod<br>tuab-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.881405 |  | 0.874224 | 0.794577 |  |
| `55243f1c850de4cbdd5f` | cbramod<br>tuab-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.886797 |  | 0.878945 | 0.801047 |  |
| `6f6357945995fcacc0cc` | cbramod<br>tuab-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.881405 |  | 0.874224 | 0.794577 |  |
| `d3858d0a3cab7d7eb655` | cbramod<br>tuab-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.882738 |  | 0.875436 | 0.799499 |  |
| `1b56a574752eb7cabbf3` | cbramod<br>tuab-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.873939 |  | 0.871654 | 0.780027 |  |
| `6821f8b82dce09d3b841` | cbramod<br>tuab-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.883114 |  | 0.877129 | 0.786870 |  |
| `a3173e2a47614baeca03` | cbramod<br>tuab-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.884737 |  | 0.876275 | 0.804167 |  |
| `d327baff6c4df28ecd91` | cbramod<br>tuab-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.884737 |  | 0.876275 | 0.804167 |  |
| `7832c50aed8a5a69dff1` | cbramod_paclockfe宿主转接<br>tuab-cbramod_crofremo | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.857879 |  | 0.861651 | 0.776970 |  |
| `a1eefc6a5973d0b30d0a` | cbramod_paclockfe宿主转接<br>tuab-cbramod_crofremo | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.858811 |  | 0.854087 | 0.773887 |  |
| `987b9cb1d0a2ff89d9f0` | cbramod_paclockfe宿主转接<br>tuab-cbramod_crofremo | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.853821 |  | 0.856251 | 0.778628 |  |
| `d63d38a419f8a7786349` | cnn_transformer<br>tuab-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.891069 |  | 0.891021 | 0.780053 |  |
| `66e9f78689fd551e9966` | cnn_transformer<br>tuab-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.898263 |  | 0.897197 | 0.821521 |  |
| `66f06631af6f4e97afdd` | cnn_transformer<br>tuab-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.873481 |  | 0.885813 | 0.743201 |  |
| `6a4c898b2cf7573bb068` | contrawr<br>tuab-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.888883 |  | 0.884923 | 0.789945 |  |
| `c10d5c58cf8f4eae54bb` | contrawr<br>tuab-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.882027 |  | 0.883876 | 0.787771 |  |
| `bd19dc09894f925f0ce7` | contrawr<br>tuab-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.897093 |  | 0.896930 | 0.790675 |  |
| `173282abc0304276ec80` | csbrain<br>tuab-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.895847 |  | 0.888959 | 0.808871 |  |
| `240bd71ca1a9cfed862e` | csbrain<br>tuab-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.898502 |  | 0.893517 | 0.800416 |  |
| `9f8a9ce93923ba9726f7` | csbrain<br>tuab-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.904536 |  | 0.900491 | 0.818646 |  |
| `79b7f49c74bc69c35680` | eegconformer<br>seed0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.882931 | 0.882934 | 0.878496 | 0.793846 |  |
| `8771ffbeef486273d050` | eegconformer<br>tuab-eegconformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.888899 |  | 0.884301 | 0.796729 |  |
| `2829f9e36befb999c520` | eegconformer<br>tuab-eegconformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.882345 |  | 0.882836 | 0.767098 |  |
| `605ffeb9bb401ee6fa7c` | eegconformer<br>tuab-eegconformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.878912 |  | 0.881327 | 0.803844 |  |
| `fcfe2989201ef51588c9` | eegconformer<br>tuab-eegconformer_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.873561 |  | 0.877180 | 0.761565 |  |
| `09c25c173db7670c8fcb` | eegconformer<br>tuab-eegconformer_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.870846 |  | 0.867152 | 0.757279 |  |
| `cf2dd5dc091221ce391d` | eegconformer<br>tuab-eegconformer_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.885736 |  | 0.881143 | 0.803237 |  |
| `ae794002b2b3c2f5e92a` | eegconformer<br>tuab-eegconformer_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.888899 |  | 0.884301 | 0.796729 |  |
| `fdf04ce59871940327f2` | eegnet<br>tuab-eegnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.851524 |  | 0.855082 | 0.771641 |  |
| `b5712f3b039c61896a3b` | eegnet<br>tuab-eegnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.854603 |  | 0.853456 | 0.769158 |  |
| `a22399d6f0d0bd0e6f9f` | eegnet<br>tuab-eegnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.847140 |  | 0.847095 | 0.763565 |  |
| `11302f96283ffb5a8f88` | eegnet<br>tuab-eegnet_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.839236 |  | 0.847999 | 0.769315 |  |
| `da5bb5b70cc16ba986db` | eegnet<br>tuab-eegnet_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.851524 |  | 0.855082 | 0.771641 |  |
| `2027b29deeadead2a4c1` | eegnet<br>tuab-eegnet_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.830888 |  | 0.841015 | 0.758212 |  |
| `4df9febcf6881ebf959c` | eegnet<br>tuab-eegnet_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.827199 |  | 0.836510 | 0.748354 |  |
| `5960a8272cf3f671f828` | eegpt<br>tuab-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.907449 |  | 0.904572 | 0.818170 |  |
| `f9a90771440105d91b7b` | eegpt<br>tuab-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.901292 |  | 0.898141 | 0.816670 |  |
| `39f6034f4c49611be9e0` | eegpt<br>tuab-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.907338 |  | 0.905693 | 0.806400 |  |
| `1ce62533a60623d43815` | feat_best<br>tuab-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.815452 |  | 0.824808 | 0.744514 |  |
| `6aea56249a21bbf0c3f7` | feat_best<br>tuab-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.815452 |  | 0.824808 | 0.744514 |  |
| `6d0faa76b1de1809495e` | feat_best<br>tuab-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.815452 |  | 0.824808 | 0.744514 |  |
| `28eef734f0624acc3582` | ffcl<br>tuab-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.887480 |  | 0.889648 | 0.725291 |  |
| `268deab5b302caebf288` | ffcl<br>tuab-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.892616 |  | 0.892879 | 0.806125 |  |
| `9886675f9bbc15e0562f` | ffcl<br>tuab-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.880318 |  | 0.876921 | 0.763057 |  |
| `6573a1b378ff1f04ea6b` | labram<br>tuab-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.840939 |  | 0.874994 | 0.797701 |  |
| `b5fcd4513de5078f1b0f` | labram<br>tuab-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.869633 |  | 0.874787 | 0.793837 |  |
| `a3db3c64b52ead874649` | labram<br>tuab-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.865276 |  | 0.881074 | 0.796215 |  |
| `7b17219d9989ac02ab9f` | labram<br>tuab-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.851468 |  | 0.861744 | 0.778126 |  |
| `ae2af09216d5652bdc36` | labram<br>tuab-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.850005 |  | 0.869533 | 0.784602 |  |
| `967e794f25c3892cb5f9` | labram<br>tuab-labram_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.868431 |  | 0.879927 | 0.797772 |  |
| `c214e9cf9e2ed12e4ac0` | labram<br>tuab-labram_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.860586 |  | 0.868225 | 0.781195 |  |
| `26651e6bb12cc383acef` | labram<br>tuab-labram_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.862990 |  | 0.875635 | 0.792673 |  |
| `42819428b5c065294ee8` | labram<br>tuab-labram_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.848016 |  | 0.854295 | 0.771438 |  |
| `ded5407a1e670abb84a4` | reve<br>tuab-reve_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.911184 |  | 0.903730 | 0.813233 |  |
| `44c9ae7e2017ea53e97b` | reve<br>tuab-reve_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.895016 |  | 0.894957 | 0.814485 |  |
| `16d83bd20b874da4fd92` | reve<br>tuab-reve_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.890886 |  | 0.884776 | 0.788763 |  |
| `032813c81d173f320d57` | reve<br>tuab-reve_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run |  | 0.846019 | 0.846018 | 0.853134 | 0.762798 |  |
| `8fccc8597a73b6b4b3e0` | sparcnet<br>tuab-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.883896 |  | 0.883594 | 0.790838 |  |
| `3ed897dacfb06dc1d0b3` | sparcnet<br>tuab-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.888727 |  | 0.889648 | 0.781811 |  |
| `0108851dcaa46b6d71a3` | sparcnet<br>tuab-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.889579 |  | 0.889809 | 0.804332 |  |
| `69cc2e16469717ca3ab7` | st_transformer<br>tuab-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.904223 |  | 0.901676 | 0.824467 |  |
| `1f3412c057395e3973ab` | st_transformer<br>tuab-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.904283 |  | 0.899209 | 0.819489 |  |
| `cb432ba6fda05107db73` | st_transformer<br>tuab-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.898047 |  | 0.890819 | 0.815560 |  |
| `01333783454822509402` | tfm<br>tuab-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.882042 |  | 0.878466 | 0.789580 |  |
| `5f014d7a9c959458f42d` | tfm<br>tuab-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.887775 |  | 0.882096 | 0.803322 |  |
| `4ddc56282e88b203928d` | tfm<br>tuab-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.882163 |  | 0.877243 | 0.795297 |  |
| `3df65df1b789e1e7dbf8` | 频带关系内容版<br>tuab-paclock_pt_base | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.882651 |  | 0.879915 | 0.802410 |  |
| `2779e2cd4c13f9ce1c3e` | 频带关系内容版<br>tuab-paclock_pt_base | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.887708 |  | 0.883758 | 0.803551 |  |
| `b1b3f2112a9461e9f09b` | 频带关系内容版<br>tuab-paclock_pt_base | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.891971 |  | 0.888479 | 0.820290 |  |
| `ec91b18b5a201a464392` | 频带关系内容版<br>tuab-paclock_pt_large | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.891713 |  | 0.883109 | 0.787201 |  |
| `b87808e9aded3adef2d8` | 频带关系内容版<br>tuab-paclock_pt_large | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.902156 |  | 0.894832 | 0.815312 |  |
| `3c42fd3f04438f23744e` | 频带关系内容版<br>tuab-paclock_pt_large | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.889043 |  | 0.882745 | 0.803613 |  |
| `ccaec5bb74e0f8ebb6de` | 频带关系内容版<br>tuab-paclock_full | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.858589 |  | 0.851602 | 0.776405 |  |
| `b24fe5fe66f973dbf175` | 频带关系内容版<br>tuab-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.881878 |  | 0.880643 | 0.799442 |  |
| `3a09391a4416b9074817` | 频带关系内容版<br>tuab-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.888775 |  | 0.885524 | 0.803812 |  |
| `2073f75726ce8c0fc7e6` | 频带关系内容版<br>tuab-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.888759 |  | 0.880210 | 0.806099 |  |
| `320fa274133e557fe4c6` | 频带关系内容版<br>tuab-paclock_v2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.893346 |  | 0.887441 | 0.810297 |  |
| `ffa51e861118afa6ef00` | 频带关系内容版<br>tuab-paclock_v2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.889730 |  | 0.881771 | 0.796351 |  |
| `c944ff4b77d8b028fa98` | 频带关系内容版<br>tuab-paclock_v2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.883914 |  | 0.879514 | 0.806950 |  |
| `8681ec91d0d805c32493` | 频带波形原始版<br>tuab-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.883905 |  | 0.881310 | 0.800147 |  |
| `e368304152b54de85711` | 频带波形原始版<br>tuab-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.885807 |  | 0.879911 | 0.803882 |  |
| `23ba98af55c4ed966376` | 频带波形原始版<br>tuab-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.897267 |  | 0.891400 | 0.812088 |  |
