# TUEV 实验记录

[返回总入口](../README.md)

共 447 条记录。[完整精度及来源 CSV](tuev.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（63 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `c72b88c0fd9da52dddc0` | biot<br>tuev-biot_scratch_native20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_checkpoint_eval | 0.418147 |  |  |  | 0.425059 | 0.696596 |
| `f473a3962deafa649ece` | biot<br>tuev-biot_scratch_native20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  |  |  |  |  |  |
| `1a9c94dcf3d5ad2ba4af` | biot<br>tuev-biot_pretrained_native20261025 | pretrained | 0 | completed_record<br>clinical_checkpoint_eval | 0.420002 |  |  |  | 0.488141 | 0.696793 |
| `e583f6f74ddd2d223788` | biot<br>tuev-biot_pretrained_native20261025 | pretrained | 0 | completed_record<br>clinical_run |  |  |  |  |  |  |
| `fb9df85e5ca631bc60b8` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.530350 |  |  |  | 0.505745 | 0.761013 |
| `fd425889259c9657fe7c` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.532391 |  |  |  | 0.516630 | 0.757428 |
| `1ba812d19f0c8563914d` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.387248 |  |  |  | 0.348695 | 0.680134 |
| `6118a34518ba3cf54163` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.356786 |  |  |  | 0.306369 | 0.680455 |
| `2f0265eac1455008c31b` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.353681 |  |  |  | 0.307370 | 0.671264 |
| `ed2dd86847a1a891309f` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.319046 |  |  |  | 0.289306 | 0.640805 |
| `26fc97a35052aec69666` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.436927 |  |  |  | 0.458233 | 0.711700 |
| `953745792d395add0da9` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.413026 |  |  |  | 0.329756 | 0.686141 |
| `0099510b9911ef6a8ff2` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.402227 |  |  |  | 0.482728 | 0.684554 |
| `b8e2151df4a3636eb5ec` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.384289 |  |  |  | 0.295729 | 0.689249 |
| `11a9bd2a664abe9cefbc` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.523630 |  |  |  | 0.509981 | 0.758300 |
| `44ca403b88ac701db24e` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.362111 |  |  |  | 0.276718 | 0.678332 |
| `280da013485cf7794f27` | eegconformer<br>tuev-eegconformer_native_s0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.502888 |  |  |  | 0.488405 | 0.740858 |
| `f665930d6c77908903b9` | eegpt<br>tuev-eegpt_pretrained_shared16_20261025 | pretrained | 0 | completed_record<br>clinical_run | 0.634257 |  |  |  | 0.594330 | 0.812125 |
| `456c205a454e35a1b01b` | eegpt<br>tuev-eegpt_scratch_native23_20261022 | scratch | 0 | completed_record<br>clinical_run | 0.489400 |  |  |  | 0.455991 | 0.753229 |
| `e6bc7c69a9f247a953b2` | eegpt<br>tuev-eegpt_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run | 0.627749 |  |  |  | 0.531496 | 0.808189 |
| `0dcc6e0f8727c4f27b31` | eegpt<br>tuev-eegpt_scratch_shared16_20261026 | scratch | 0 | completed_record<br>clinical_run | 0.577934 |  |  |  | 0.529786 | 0.783212 |
| `9f4cd58f8e543b63d2a0` | feat_best<br>tuev-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.504931 |  |  |  | 0.448109 | 0.744545 |
| `6aba527571ed6adfff6c` | feat_best<br>tuev-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.504931 |  |  |  | 0.448109 | 0.744545 |
| `2ae39ec59d2e2d221388` | feat_best<br>tuev-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.504931 |  |  |  | 0.448109 | 0.744545 |
| `2ece1bb9c87277db45ac` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.558530 |  |  |  | 0.404737 | 0.758335 |
| `a65d8d836c7bec5f63be` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.507864 |  |  |  | 0.525086 | 0.747226 |
| `5673801c931e2960d4c7` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.400754 |  |  |  | 0.321548 | 0.686159 |
| `c5fc4c2afd7ee0d3eaca` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.364209 |  |  |  | 0.373386 | 0.676077 |
| `7f5ccd849648a81cc3b8` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.487739 |  |  |  | 0.434797 | 0.739559 |
| `cdb5ab038e3cce3ad691` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.424832 |  |  |  | 0.322753 | 0.697425 |
| `ca7bf28584d20684cd4a` | ffcl<br>tuev-ffcl_native20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  |  |  |  |  |  |
| `eadc208046e70077fe59` | ffcl<br>tuev-ffcl_native20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_checkpoint_eval | 0.458560 |  |  |  | 0.393866 | 0.722813 |
| `46652f98cd52ad2908e8` | labram<br>tuev-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.601204 |  |  |  | 0.615830 | 0.805935 |
| `71d89a35219d0a86e9d9` | labram<br>tuev-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.554972 |  |  |  | 0.587236 | 0.784560 |
| `9ec618692763e09a3050` | labram<br>tuev-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.424024 |  |  |  | 0.506564 | 0.711322 |
| `004831eded2520b97c7c` | labram<br>tuev-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.598926 |  |  |  | 0.609988 | 0.814142 |
| `06814fa6a1cf42ba2e2f` | labram<br>tuev-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.425491 |  |  |  | 0.465752 | 0.727334 |
| `cd9dfc04ce0b5a09b5f6` | labram<br>tuev-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.634879 |  |  |  | 0.667197 | 0.826761 |
| `085b62e83b4651f20c6e` | labram<br>tuev-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.389547 |  |  |  | 0.479293 | 0.694824 |
| `1921bd42a0112968cd41` | labram<br>tuev-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.660760 |  |  |  | 0.676372 | 0.834218 |
| `c621f2743de0388a501f` | labram<br>tuev-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.595115 |  |  |  | 0.665644 | 0.805853 |
| `1b58db3ad039a60f966b` | labram<br>tuev-labram_scratch | scratch | 0 | completed_record<br>clinical_run | 0.362419 |  |  |  | 0.446985 | 0.678595 |
| `7efd2b4940fa2a0743ad` | labram<br>tuev-labram_scratch | scratch | 0 | completed_record<br>clinical_run | 0.355209 |  |  |  | 0.404086 | 0.682020 |
| `89881ad87f27314f024c` | labram<br>tuev-labram_scratch | scratch | 1 | completed_record<br>clinical_run | 0.372701 |  |  |  | 0.279294 | 0.704893 |
| `df236751bdb5e5900053` | labram<br>tuev-labram_scratch | scratch | 1 | completed_record<br>clinical_run | 0.353309 |  |  |  | 0.382088 | 0.679209 |
| `84cddcdfdee4c6706c28` | labram<br>tuev-labram_scratch | scratch | 2 | completed_record<br>clinical_run | 0.312741 |  |  |  | 0.404206 | 0.657432 |
| `8c592bd45851ff002d06` | labram<br>tuev-labram_scratch | scratch | 2 | completed_record<br>clinical_run | 0.380107 |  |  |  | 0.390772 | 0.694790 |
| `7ef6154d09061dfd8960` | labram_paclockfe宿主转接<br>tuev-labram_crofremo | scratch | 0 | completed_record<br>clinical_run | 0.521999 |  |  |  | 0.373438 | 0.752305 |
| `3fa7f7d0aed5655b1879` | labram_paclockfe宿主转接<br>tuev-labram_crofremo | scratch | 1 | completed_record<br>clinical_run | 0.482660 |  |  |  | 0.391441 | 0.721342 |
| `729d5989c2d38d95049c` | labram_paclockfe宿主转接<br>tuev-labram_crofremo | scratch | 2 | completed_record<br>clinical_run | 0.502682 |  |  |  | 0.371540 | 0.731327 |
| `ffafd04e8c239187cace` | reve<br>tuev-reve_pretrained_rebuild20261021 | pretrained | 0 | completed_record<br>clinical_run | 0.582950 |  |  |  | 0.532561 | 0.784309 |
| `c2790a04970af43851c8` | reve<br>tuev-reve_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run | 0.423435 |  |  |  | 0.440503 | 0.694548 |
| `a6287c4a4693568e47cb` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.372660 |  |  |  | 0.346349 | 0.687504 |
| `b0580c91c82e6fec85cf` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.539825 |  |  |  | 0.405844 | 0.760113 |
| `61c9085d2c78e84f8c4d` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.431363 |  |  |  | 0.432150 | 0.707838 |
| `e567742a7e6c1bc20b80` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.275538 |  |  |  | 0.331282 | 0.639179 |
| `45cca3ed9102dc061fae` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.523169 |  |  |  | 0.452229 | 0.753148 |
| `87bad2a521d7bcec9d4a` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.452735 |  |  |  | 0.416152 | 0.696006 |
| `96ffdc2880fce77a983c` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.480675 |  |  |  | 0.431473 | 0.731535 |
| `a491a2c0e560d018b798` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.516757 |  |  |  | 0.456923 | 0.742765 |
| `42585e9d7d4c4d3cc006` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.484781 |  |  |  | 0.439485 | 0.714926 |
| `b100705fff8141f10372` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.479248 |  |  |  | 0.457732 | 0.731769 |
| `7122d48bd073c920f2b5` | tfm<br>tuev-tfm_single_rebuild20261021 | pretrained | 0 | completed_record<br>clinical_run | 0.599408 |  |  |  | 0.567544 | 0.790529 |

## shared16_29421（382 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `e6f294c75a922850317b` | CBraMod前端 → CfFT双轴<br>tuev-cbramod_tokenizer_twoaxis_20261021 | scratch | 0 | completed_record<br>clinical_run | 0.738096 |  |  |  | 0.688659 | 0.862209 |
| `80ba96b1cc023e1d1146` | CFC参考替换<br>tuev-cfft_cfc_reference_20261030 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.703982 |  |  |  | 0.627238 | 0.841839 |
| `2069a1e966ee454baa45` | Candidate C 单路径<br>tuev-cfft_direct_20261104 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.660108 |  |  |  | 0.642785 | 0.826950 |
| `deaafed29bc9f0e5b4d8` | CfFT其他前端<br>tuev-cfft_copula_v1_lenovo_20261025 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.651209 |  |  |  | 0.573670 | 0.816637 |
| `915f703ddc81f2776d45` | CfFT其他前端<br>tuev-factorized_f0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.672011 |  |  |  | 0.574911 | 0.828221 |
| `377c19b8719eb6caf7a1` | CfFT其他前端<br>tuev-factorized_f1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.689182 |  |  |  | 0.665428 | 0.839412 |
| `3915b0e674019b41d0fb` | CfFT其他前端<br>tuev-factorized_f1_scale03_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.696553 |  |  |  | 0.541525 | 0.836780 |
| `d20ca0e87dd6b9850951` | CfFT其他前端<br>tuev-factorized_f1_scale03_mi2101x | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.696912 |  |  |  | 0.657824 | 0.841327 |
| `6f20b5b6ab15d466cef1` | CfFT其他前端<br>tuev-factorized_f1_scale1_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.658097 |  |  |  | 0.637959 | 0.822305 |
| `89206e91f9a0a0889dea` | CfFT其他前端<br>tuev-factorized_f1_scale1_mi2101x | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.684772 |  |  |  | 0.653466 | 0.833852 |
| `e42c81469bcdc68b9a36` | CfFT其他前端<br>tuev-factorized_f2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.691179 |  |  |  | 0.660756 | 0.838503 |
| `2cd7f8ae75a11b8e13f1` | CfFT其他前端<br>tuev-factorized_f4_aug_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.701168 |  |  |  | 0.624726 | 0.840581 |
| `2b39a9879a1a57715e2d` | CfFT其他前端<br>tuev-factorized_f4_blockcontrol | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.594980 |  |  |  | 0.569875 | 0.788194 |
| `c3bd469de7c437763acd` | CfFT其他前端<br>tuev-factorized_f4_joint_confirm | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.712655 |  |  |  | 0.633917 | 0.844866 |
| `75567438d7ab11e8a221` | CfFT其他前端<br>tuev-factorized_f4_rmsgeglu | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.681579 |  |  |  | 0.681487 | 0.835071 |
| `122f348e2596d3845429` | CfFT其他前端<br>tuev-factorized_f4_rmsgeglu | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.675239 |  |  |  | 0.623083 | 0.822679 |
| `ba4d568f5c949b5792c0` | CfFT其他前端<br>tuev-paclock_fuse | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.549304 |  |  |  | 0.493231 | 0.763983 |
| `0038127ba04b823b0f57` | CfFT其他前端<br>tuev-paclock_fusegate | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.587947 |  |  |  | 0.551562 | 0.788837 |
| `9b871b4405e948af4899` | CfFT其他前端<br>tuev-paclock_fusegate | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.524970 |  |  |  | 0.456812 | 0.750121 |
| `f4dff6fd3cd2266b76e7` | CfFT其他前端<br>tuev-paclock_fusegate | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.564391 |  |  |  | 0.481490 | 0.776321 |
| `89689ba250ea1b43d292` | CfFT其他前端<br>tuev-paclock_hybrid | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.695107 |  |  |  | 0.640975 | 0.836478 |
| `01f2ed8c755a73f7943a` | CfFT其他前端<br>tuev-paclock_hybrid_attn | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.681362 |  |  |  | 0.530914 | 0.829187 |
| `24213924102242d2609e` | CfFT其他前端<br>tuev-paclock_hybrid_gate | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.692855 |  |  |  | 0.624410 | 0.837458 |
| `33104e45b1da55c3cb86` | CfFT其他前端<br>tuev-paclock_hybrid_gate | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.714090 |  |  |  | 0.528800 | 0.841139 |
| `194f8f77642e9d7ab9e2` | CfFT其他前端<br>tuev-paclock_hybrid_gate | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.611734 |  |  |  | 0.516685 | 0.796140 |
| `d8d37301a2059784b5d5` | Duplex<br>tuev-cf2_v0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.658364 |  |  |  | 0.530120 | 0.820895 |
| `5459377292707a8cd8be` | Duplex<br>tuev-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.689561 |  |  |  | 0.595678 | 0.833798 |
| `2f51abf03ac68fa4b857` | Duplex<br>tuev-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.720666 |  |  |  | 0.653978 | 0.849154 |
| `5d7ef1e221a75510c1ed` | Duplex<br>tuev-cf2_v1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.647831 |  |  |  | 0.625224 | 0.818105 |
| `26c1efda7fe6a13635e9` | Duplex<br>tuev-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.679195 |  |  |  | 0.522447 | 0.825948 |
| `d9d6b6b082b158cc48cd` | Duplex<br>tuev-cf2_v1d256 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.666118 |  |  |  | 0.659281 | 0.824417 |
| `fe027e37919a83baf441` | Duplex<br>tuev-cf2_v2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.657300 |  |  |  | 0.635353 | 0.823153 |
| `e455027c8b77c6a6c264` | Duplex<br>tuev-paclock_flagship | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.692276 |  |  |  | 0.601532 | 0.835524 |
| `47bc4a628e70be80c465` | Duplex-A128<br>tuev-paclock_duplex_p200ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.530619 |  |  |  | 0.552214 | 0.750643 |
| `ee9c2f5873e5e990de31` | Duplex-A128<br>tuev-paclock_duplex_pt | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.689147 |  |  |  | 0.585943 | 0.832109 |
| `1264ea4af69a302048ad` | Duplex-A128<br>tuev-paclock_duplex_pt2 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.653308 |  |  |  | 0.543177 | 0.813126 |
| `23198ef7d0dfc3b99a9f` | Duplex-A128<br>tuev-paclock_duplex_pt3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.675349 |  |  |  | 0.593984 | 0.827917 |
| `549e00e4eb2a9b31c66d` | Duplex-A128<br>tuev-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.566221 |  |  |  | 0.577750 | 0.767631 |
| `532bcd0023e5d45dd790` | Duplex-A128<br>tuev-paclock_duplex_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.635737 |  |  |  | 0.619742 | 0.804578 |
| `68ca0d173883dbb630d5` | Duplex-A128<br>tuev-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.641696 |  |  |  | 0.607872 | 0.809981 |
| `e28bbf499fddcc450265` | Duplex-A128<br>tuev-paclock_lf05_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.520581 |  |  |  | 0.474998 | 0.745657 |
| `e5adc7dddc33445ce3c0` | Duplex-A128<br>tuev-paclock_lf10_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.574705 |  |  |  | 0.488352 | 0.773409 |
| `63cabaf89e8118602b3f` | Duplex-A128<br>tuev-paclock_lf25_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.598866 |  |  |  | 0.628342 | 0.788077 |
| `dd92ec5713830a40701d` | Duplex-A128<br>tuev-paclock_probe_v1 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.635095 |  |  |  | 0.456308 | 0.805819 |
| `73f3b5842d3a421f9583` | Duplex-A128<br>tuev-paclock_probe_v2 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.518450 |  |  |  | 0.399339 | 0.748830 |
| `79587c02ded4f2cac73f` | Duplex-A128<br>tuev-paclock_ptF_frozenL3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.650988 |  |  |  | 0.485822 | 0.808281 |
| `aca731ae7159619ced35` | Duplex-A128<br>tuev-paclock_v3traj_100000frozen | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.514424 |  |  |  | 0.380555 | 0.745176 |
| `4f74f37d4676f0d6f773` | Duplex-A128<br>tuev-paclock_v3traj_100000frozenL3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.563923 |  |  |  | 0.422778 | 0.765197 |
| `ad61dbbef9c739f77829` | Duplex-A128<br>tuev-paclock_v3traj_20000frozen | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.471575 |  |  |  | 0.345225 | 0.710250 |
| `ac6c14c21b4c6560507f` | Duplex-A128<br>tuev-paclock_v3traj_20000frozenL3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.539183 |  |  |  | 0.431862 | 0.752645 |
| `eff2a914c080de2ecfa8` | Duplex-A128<br>tuev-paclock_v3traj_40000frozen | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.370869 |  |  |  | 0.256554 | 0.672433 |
| `fb661989d6b3bd1367f5` | Duplex-A128<br>tuev-paclock_v3traj_40000frozenL3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.549636 |  |  |  | 0.382275 | 0.754539 |
| `872f3caa8afd21a137e2` | Duplex-A128<br>tuev-paclock_v3traj_60000frozen | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.487533 |  |  |  | 0.299255 | 0.701632 |
| `c34f8f73d8ddf53484b8` | Duplex-A128<br>tuev-paclock_v3traj_60000frozenL3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.470896 |  |  |  | 0.300535 | 0.697171 |
| `eae6e40ba0795c044d28` | Duplex-A128<br>tuev-paclock_v3traj_80000frozen | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.515555 |  |  |  | 0.346812 | 0.732576 |
| `90f55aa1559fcc4a35d6` | Duplex-A128<br>tuev-paclock_v3traj_80000frozenL3 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.531971 |  |  |  | 0.415457 | 0.754749 |
| `164c66b742bc3c560ea3` | Duplex-A128<br>tuev-a128_r1_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.718993 |  |  |  | 0.551546 | 0.846058 |
| `c139378f757724463588` | Duplex-A128<br>tuev-crofremo_s2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.702053 |  |  |  | 0.638465 | 0.841022 |
| `480257b768d309e8956c` | Duplex-A128<br>tuev-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.709401 |  |  |  | 0.646369 | 0.842827 |
| `b6e279e99ab413a47f37` | Duplex-A128<br>tuev-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.710870 |  |  |  | 0.526850 | 0.839430 |
| `67187fef54080d673244` | Duplex-A128<br>tuev-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.648375 |  |  |  | 0.535274 | 0.813741 |
| `007131e449e7f28fbd9a` | Duplex-A128<br>tuev-paclock_duplex_ms | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.705769 |  |  |  | 0.583129 | 0.839259 |
| `bcf5cb0bdc9333d2d447` | Duplex-A128<br>tuev-paclock_duplex_p200 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.609435 |  |  |  | 0.573728 | 0.795379 |
| `ad5eebf08668c369af01` | Duplex-A128<br>tuev-paclock_lf05_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.541160 |  |  |  | 0.439616 | 0.754394 |
| `f7b124f72acb42c1cae6` | Duplex-A128<br>tuev-paclock_lf10_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.693614 |  |  |  | 0.565772 | 0.836030 |
| `8ef8aabe45942098d503` | Duplex-A128<br>tuev-paclock_lf25_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.730483 |  |  |  | 0.615930 | 0.854491 |
| `223c628afaf691bdf000` | Duplex-A128<br>tuev-paclock_probe_rand | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.385937 |  |  |  | 0.269459 | 0.676028 |
| `4aa6424d01ae15b4d63a` | Duplex-F192<br>tuev-cf2_v1d192_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.640565 |  |  |  | 0.548574 | 0.807196 |
| `70e6523051ca4bb55ba3` | Duplex-F192<br>tuev-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.680414 |  |  |  | 0.583112 | 0.833759 |
| `7d351f3f4e26b9fdf8f0` | Duplex-F192<br>tuev-cf2_v1d192 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.628743 |  |  |  | 0.588538 | 0.800040 |
| `3ee3900b3bde488edfb0` | Duplex-F192<br>tuev-cf2_v1d192_r1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.657775 |  |  |  | 0.615667 | 0.820028 |
| `aae7be73b4e80ad50f2a` | Duplex-F192<br>tuev-cf2_v1d192_r2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.689420 |  |  |  | 0.658713 | 0.838406 |
| `25c2a7bae7292bae7c67` | Duplex-F192<br>tuev-cf2_v1d192_r3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.640719 |  |  |  | 0.490061 | 0.807060 |
| `ba0e5d0161520e247f5b` | Duplex-F192<br>tuev-cf2_v1d192_r4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.586927 |  |  |  | 0.494518 | 0.773228 |
| `c4cbf9a4d9c01f636fae` | Duplex-F192<br>tuev-cf2_v3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.660592 |  |  |  | 0.537082 | 0.817736 |
| `24184aca062230debb0a` | Duplex-F192 / own<br>tuev-cf2_v1d192own | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.577807 |  |  |  | 0.605442 | 0.778155 |
| `99f5e936f0ff2a19d35d` | Duplex-F192 / own<br>tuev-cf2_v1d192own | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.605167 |  |  |  | 0.612408 | 0.795730 |
| `37c71e6933d08d5e489f` | Duplex-F192 / own<br>tuev-cf2_v1d192own | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.587209 |  |  |  | 0.551147 | 0.780926 |
| `53d07a10fd43943104a0` | Quad-Range<br>tuev-cfft_twoaxis_noaug_rebuild20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.684564 |  |  |  | 0.553740 | 0.833727 |
| `ad7761798165111082e2` | Quad-Range<br>tuev-cfft_twoaxis_rebuild20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.693214 |  |  |  | 0.612351 | 0.838390 |
| `5c4052f60fd597c4e19b` | Quad-Range<br>tuev-quad_range_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.713880 |  |  |  | 0.680590 | 0.850620 |
| `e9490cee8541b96a9120` | Quad-Range<br>tuev-quad_range_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.715456 |  |  |  | 0.608749 | 0.846032 |
| `5b531da52c9b493ef09c` | Quad-Range<br>tuev-quad_range_Z1s_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.738711 |  |  |  | 0.626813 | 0.860100 |
| `6218fc0cefbf3cf376ae` | Quad-Range<br>tuev-quad_range_Z5s_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.736795 |  |  |  | 0.604176 | 0.858820 |
| `491c3b552f8c14fbfcad` | Quad-Range<br>tuev-range_twoaxis_measured_20260917 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.684601 |  |  |  | 0.636580 | 0.835755 |
| `930fc363622fe0f4f996` | Quad-Range<br>tuev-range_twoaxis_measured_20260917 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.698526 |  |  |  | 0.578195 | 0.838584 |
| `55ab79f81fa6a6377b3e` | Quad-Range<br>tuev-range_twoaxis_measured_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.713350 |  |  |  | 0.640409 | 0.851335 |
| `0035fa1c6db4a0bbf446` | Quad-Range<br>tuev-range_twoaxis_measured_m200_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.706164 |  |  |  | 0.635669 | 0.841238 |
| `ae8220545f4e9e44a635` | Quad-Range<br>tuev-range_twoaxis_measured_poolangle_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.682492 |  |  |  | 0.640065 | 0.836696 |
| `4dc13fd17715cc6b15ec` | Quad-Range / envelope-only<br>tuev-quad_range_envelope_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.713653 |  |  |  | 0.596117 | 0.844064 |
| `04d318392c5f3a6b3460` | Quad-Range / envelope-only<br>tuev-range_twoaxis_envelope_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.703625 |  |  |  | 0.626928 | 0.842453 |
| `689ece0d9e86aab77ef0` | Quad-Range / identity<br>tuev-capacity2-paired-20261106 | not_explicitly_recorded | 2 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `62fedc007f1e3de6b8f4` | Quad-Range / identity<br>tuev-demodulation-bandmix-restart-20261106 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.585392 |  |  |  | 0.529583 | 0.788173 |
| `7f6d4ad7f1ac72d9da1a` | Quad-Range / identity<br>tuev-demodulation-patchmean-20261106 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.695371 |  |  |  | 0.612483 | 0.841554 |
| `ac537b013c5de276492e` | Quad-Range / identity<br>tuev-legacy-triaxial-identity-20261105 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.711191 |  |  |  | 0.633590 | 0.849447 |
| `320e2024a0c312695954` | Quad-Range / identity<br>tuev-range_identity_sean | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.729197 |  |  |  | 0.603662 | 0.855621 |
| `b9ede9479bf51f079902` | Quad-Range / identity<br>tuev-range_twoaxis_identity_20260917 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.616390 |  |  |  | 0.585980 | 0.804667 |
| `b2b022d79e77cb132bfd` | Quad-Range / identity<br>tuev-range_twoaxis_identity_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.699077 |  |  |  | 0.563660 | 0.838002 |
| `80a4a8866e55c2d5b7bf` | Quad-Range / identity, content:envelope<br>tuev-demodulation-envelope-20261106 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.641651 |  |  |  | 0.577092 | 0.812231 |
| `908784c2ec410f66d7d7` | Quad-Range / identity, content:waveform<br>tuev-capacity2-waveform-20261106 | not_explicitly_recorded | 2 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `11bf9fa5aa499028a2c4` | Quad-Range / identity, content:waveform<br>tuev-demodulation-waveform-20261106 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.498353 |  |  |  | 0.510323 | 0.734806 |
| `07b175be37d5c1780ffb` | Quad-Range / magnitude<br>tuev-quad_range_weight_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.642524 |  |  |  | 0.600283 | 0.811686 |
| `195d7f4cb940a1233a26` | Quad-Range / magnitude<br>tuev-range_twoaxis_weight_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.661796 |  |  |  | 0.572853 | 0.813049 |
| `ec7b473bd19531a3a9b4` | Quad-Range / own<br>tuev-quad_range_own_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.532453 |  |  |  | 0.499124 | 0.754741 |
| `abd035fcc31537b9e622` | Quad-Range / own<br>tuev-range_twoaxis_own_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.600508 |  |  |  | 0.531113 | 0.789420 |
| `1f5eba840366ce4e8b5e` | Quad-Range / sample_shuffle<br>tuev-range_sample_shuffle_sean | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.600944 |  |  |  | 0.486017 | 0.786410 |
| `7906bacad98bfdf75b0d` | Quad16<br>tuev-nb16_quadrature_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.727893 |  |  |  | 0.708312 | 0.858754 |
| `51cc0a4d637d3f59d276` | Quad16<br>tuev-nb16_quadrature_20260915 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.695187 |  |  |  | 0.659277 | 0.839081 |
| `b0cdfe9e6447391e491d` | Quad16<br>tuev-nb16_quadrature_20260915 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.656620 |  |  |  | 0.600837 | 0.808105 |
| `1b48eb22007145e2a62d` | WavePatch 原波形双轴<br>tuev-wave_patch_twoaxis_noaug_rebuild20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.638717 |  |  |  | 0.506837 | 0.808384 |
| `67a2c1b7f63b6e9fb6d8` | WavePatch 原波形双轴<br>tuev-wave_patch_twoaxis_rebuild20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.637828 |  |  |  | 0.579731 | 0.814801 |
| `8693ac595c44bd2d09f4` | WavePatch 学习内容行三轴<br>tuev-wave_patch_threeaxis-20261105 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.575805 |  |  |  | 0.552468 | 0.782478 |
| `77c5b3367c3e881d6148` | a128前端 → biot<br>tuev-biot_a128_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.569221 |  |  |  | 0.515662 | 0.777803 |
| `fce04d2da46ff9ac1e8f` | a128前端 → biot<br>tuev-biot_a128_20260916 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.632947 |  |  |  | 0.486650 | 0.798890 |
| `111a124d6530e7862c01` | a128前端 → cbramod<br>tuev-cbramod_a128_20260916 | scratch | 0 | completed_record<br>clinical_run | 0.617126 |  |  |  | 0.577298 | 0.801430 |
| `5e582148075bcb0e2c30` | biot<br>tuev-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.500537 |  |  |  | 0.461218 | 0.738276 |
| `03a4f71925369f0cfd34` | biot<br>tuev-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.535378 |  |  |  | 0.485723 | 0.751767 |
| `2a3f4d4dcb1b8a6114f6` | biot<br>tuev-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.555353 |  |  |  | 0.521270 | 0.769706 |
| `78342c19482b129e8b4c` | biot<br>tuev-biot_hop50 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.566340 |  |  |  | 0.521424 | 0.777743 |
| `264c3a96fa86bc0b45d8` | biot<br>tuev-biot_hop50 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.539689 |  |  |  | 0.488012 | 0.767224 |
| `cd4e9f7f1b124d25c133` | biot<br>tuev-biot_hop50 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.493603 |  |  |  | 0.496958 | 0.733856 |
| `30c48235f7baea652ca0` | biot<br>tuev-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.503869 |  |  |  | 0.502116 | 0.747094 |
| `038caeea9887fc2abeb6` | biot<br>tuev-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.542502 |  |  |  | 0.607127 | 0.764970 |
| `7f9b723b1a9837fdeafc` | biot<br>tuev-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.550787 |  |  |  | 0.496045 | 0.771396 |
| `254a4dbe809fb797249e` | biot<br>tuev-biot_scratch16_rebuild20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.417103 |  |  |  | 0.443800 | 0.694115 |
| `0114044f4f1f621ec7d5` | biot<br>tuev-biot_scratch_rebuild20261021 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.412650 |  |  |  | 0.468809 | 0.693818 |
| `01a1c1ff017f68d964ec` | biot<br>tuev-biot_tok100 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.488023 |  |  |  | 0.588627 | 0.728794 |
| `923a6f6e393d32dffcf6` | biot<br>tuev-biot_tok100 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.532813 |  |  |  | 0.508042 | 0.758004 |
| `749917d707e62fc8bf0b` | biot<br>tuev-biot_tok100 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.412552 |  |  |  | 0.454720 | 0.691015 |
| `a39f4b605324eba68bd5` | biot_tfm_host宿主转接<br>tuev-biot_tfm_host_20260918 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.436473 |  |  |  | 0.430180 | 0.696731 |
| `11490c20ca280e872636` | biot_tfm_host宿主转接<br>tuev-biot_tfm_host_20260918 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.471539 |  |  |  | 0.441394 | 0.720722 |
| `74eaf6aae711f89121b6` | biot_tfm_host宿主转接<br>tuev-biot_tfm_host_20260918 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.437702 |  |  |  | 0.414573 | 0.705584 |
| `2f86caa966ca997d7c0b` | cbramod<br>tuev-cbramod_pac | pretrained | 0 | completed_record<br>clinical_run | 0.566569 |  |  |  | 0.545498 | 0.777018 |
| `3157407271ba46946eda` | cbramod<br>tuev-cbramod_pac | pretrained | 1 | completed_record<br>clinical_run | 0.556150 |  |  |  | 0.589417 | 0.768941 |
| `ddbddbc5056eafac8da9` | cbramod<br>tuev-cbramod_pac | pretrained | 2 | completed_record<br>clinical_run | 0.565071 |  |  |  | 0.555249 | 0.776168 |
| `5304b8933bbc62d29365` | cbramod<br>tuev-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.522554 |  |  |  | 0.495001 | 0.751924 |
| `6dbe730a4cd605d99cb7` | cbramod<br>tuev-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.477867 |  |  |  | 0.511718 | 0.727744 |
| `ea7418e13c0a1d13901a` | cbramod<br>tuev-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.516958 |  |  |  | 0.432030 | 0.752351 |
| `f36548147226884cb5cd` | cbramod<br>tuev-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.619876 |  |  |  | 0.615171 | 0.800017 |
| `44f803b12ad584e48ad8` | cbramod<br>tuev-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.656563 |  |  |  | 0.598805 | 0.822675 |
| `650230778c1993d9c61d` | cbramod<br>tuev-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.610426 |  |  |  | 0.501718 | 0.794517 |
| `9042f6bb15e176c2ef29` | cbramod<br>tuev-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.610426 |  |  |  | 0.501718 | 0.794517 |
| `d553ef893f31ae48fcf0` | cbramod<br>tuev-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.515061 |  |  |  | 0.549063 | 0.746366 |
| `1cfe4e6b3556b9da0c71` | cbramod<br>tuev-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.658174 |  |  |  | 0.594825 | 0.816368 |
| `1db92a7fc3a78692607c` | cbramod<br>tuev-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.633368 |  |  |  | 0.616078 | 0.806192 |
| `874cd1f82b1584f89d25` | cbramod<br>tuev-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.663527 |  |  |  | 0.612272 | 0.820858 |
| `f810848d9ee045a9dbf1` | cbramod<br>tuev-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.529272 |  |  |  | 0.501209 | 0.752140 |
| `340c8a7a3a300e21849d` | cbramod<br>tuev-cbramod_pretrained_head800_confirm | pretrained | 1 | completed_record<br>clinical_run | 0.656563 |  |  |  | 0.598805 | 0.822675 |
| `a6b6d660bed5bfd99dea` | cbramod<br>tuev-cbramod_pretrained_nativehead | pretrained | 1 | completed_record<br>clinical_run | 0.669166 |  |  |  | 0.617513 | 0.824679 |
| `7c1cdbcfd54454bca9da` | cbramod<br>tuev-cbramod_ptn_ds4 | scratch | 0 | completed_record<br>clinical_run | 0.493153 |  |  |  | 0.528617 | 0.739705 |
| `6cea04ace432080d782e` | cbramod<br>tuev-cbramod_ptn_tuegc | scratch | 0 | completed_record<br>clinical_run | 0.500423 |  |  |  | 0.531057 | 0.746238 |
| `14e8029ae29e86e3f0d1` | cbramod<br>tuev-cbramod_ptn_tuegc_140k_gate | scratch | 0 | completed_record<br>clinical_run | 0.513606 |  |  |  | 0.509117 | 0.755583 |
| `1ecd58d5b912b655b6df` | cbramod<br>tuev-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.560823 |  |  |  | 0.497004 | 0.769924 |
| `3c2241c7f4a447afac95` | cbramod<br>tuev-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.512059 |  |  |  | 0.530283 | 0.746622 |
| `ad9d43d4ef7661b52e51` | cbramod<br>tuev-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.600813 |  |  |  | 0.560412 | 0.791408 |
| `db929df3cc0f104c0951` | cbramod<br>tuev-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.575503 |  |  |  | 0.586034 | 0.771890 |
| `3c6bf3e02b89fed8e0a9` | cbramod<br>tuev-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.638303 |  |  |  | 0.579936 | 0.809332 |
| `486e563b52cfbbb4828d` | cbramod<br>tuev-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.547882 |  |  |  | 0.566034 | 0.764947 |
| `837a4cfded37e60153c6` | cbramod<br>tuev-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.638303 |  |  |  | 0.579936 | 0.809332 |
| `de9784b3a715dba4b597` | cbramod<br>tuev-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.536637 |  |  |  | 0.565736 | 0.764620 |
| `1788fa7bc27b523c7473` | cbramod<br>tuev-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.579372 |  |  |  | 0.571362 | 0.764828 |
| `692cf757798a24feb43c` | cbramod<br>tuev-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.630678 |  |  |  | 0.583276 | 0.800376 |
| `c0c48dcc0649c6f3e4b1` | cbramod<br>tuev-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.630344 |  |  |  | 0.595371 | 0.802831 |
| `ee837315d0dfac1413c2` | cbramod<br>tuev-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.550145 |  |  |  | 0.505361 | 0.766026 |
| `98bf64037a2861a3c4ae` | cbramod<br>tuev-cbramod_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run | 0.580575 |  |  |  | 0.601747 | 0.768177 |
| `8aa1ff78da7fbdaf8601` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_cpl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.640652 |  |  |  | 0.601913 | 0.806170 |
| `892d30d79f07273507ab` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_cpl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.653596 |  |  |  | 0.616938 | 0.811985 |
| `9301e09fd193bb5d8a63` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_cpl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.607265 |  |  |  | 0.589915 | 0.795684 |
| `c96b9083065fd1c013f9` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_cplmean | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.641905 |  |  |  | 0.583850 | 0.815412 |
| `b92f3451b71f885d2ca2` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.595559 |  |  |  | 0.607505 | 0.789715 |
| `c711a90fe7f2424f5a83` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_raw | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.576041 |  |  |  | 0.603317 | 0.778090 |
| `7b14054daeb6653edd7f` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_raw | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.580999 |  |  |  | 0.560331 | 0.781014 |
| `94011c2af2e3e6a89ae0` | cbramod_paclockfe宿主转接<br>tuev-cbramod_crofremo | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.616041 |  |  |  | 0.622034 | 0.787070 |
| `93cef8923fbe97ce4758` | cbramod_paclockfe宿主转接<br>tuev-cbramod_crofremo | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.606486 |  |  |  | 0.573530 | 0.784736 |
| `fd5912864f19acfa9e30` | cbramod_paclockfe宿主转接<br>tuev-cbramod_crofremo | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.611998 |  |  |  | 0.586382 | 0.792534 |
| `764c2ee16b80f1b08468` | cbramod_paclockfe宿主转接<br>tuev-cbramod_crofremo_bands | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.632207 |  |  |  | 0.620411 | 0.804467 |
| `a3f9a1dbffc8f835716a` | cbramod_paclockfe宿主转接<br>tuev-cbramod_crofremo_bands | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.628057 |  |  |  | 0.553502 | 0.800264 |
| `c370b5354f4a349e6a7b` | cbramod_paclockfe宿主转接<br>tuev-cbramod_crofremo_bands | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.605357 |  |  |  | 0.600117 | 0.787098 |
| `fdcc7dcebedb4e2594b0` | cbramod_paclockfe宿主转接<br>tuev-cbramod_pacfe_pac | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.627953 |  |  |  | 0.571944 | 0.800824 |
| `43656b347fe2393690f6` | cbramod_paclockfe宿主转接<br>tuev-cbramod_pacfe_raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.604032 |  |  |  | 0.572436 | 0.785610 |
| `ece74061d434df304559` | cbramod_paclockfe宿主转接<br>tuev-cbramod_paclockfe | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.627953 |  |  |  | 0.571944 | 0.800824 |
| `0d0348bae0a2e97c2f17` | cbramod_paclockfe宿主转接<br>tuev-cbramod_rawfe_bands | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.595332 |  |  |  | 0.572522 | 0.783928 |
| `1a0095c615dd04c7d80a` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_cplpre | pretrained | 0 | completed_record<br>clinical_run | 0.623277 |  |  |  | 0.665777 | 0.804370 |
| `a6797db2008f966a3e05` | cbramod_paclockfe宿主转接<br>tuev-cbramod_add_rawpre | pretrained | 0 | completed_record<br>clinical_run | 0.571922 |  |  |  | 0.594915 | 0.775489 |
| `53e136797eb2814b034b` | cbramod_paclockfe宿主转接<br>tuev-cbramod_ptc_ds4 | scratch | 0 | completed_record<br>clinical_run | 0.653582 |  |  |  | 0.569332 | 0.821135 |
| `37bcb4ed12970e6349e7` | cbramod_paclockfe宿主转接<br>tuev-cbramod_ptc_tuegc_140k_gate | scratch | 0 | completed_record<br>clinical_run | 0.585501 |  |  |  | 0.565609 | 0.787852 |
| `3d0d67fab8e739746156` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.498667 |  |  |  | 0.475257 | 0.746202 |
| `55d0c45bc5d1d6cb172c` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.277899 |  |  |  | 0.270732 | 0.606947 |
| `c65337c8fdd4ab0d59a3` | cnn_transformer<br>tuev-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.452743 |  |  |  | 0.357728 | 0.721795 |
| `cd69f1efeabfa374195a` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.440993 |  |  |  | 0.476611 | 0.713669 |
| `6c6cdeddf36aa3c6d13b` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.484867 |  |  |  | 0.506639 | 0.730350 |
| `164e53e85e245586f3d9` | contrawr<br>tuev-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.296851 |  |  |  | 0.275311 | 0.644562 |
| `201e5bb3ba2bad06f070` | csbrain<br>tuev-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.619062 |  |  |  | 0.613551 | 0.803316 |
| `20f30a6ff194f2c5dbc6` | csbrain<br>tuev-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.613207 |  |  |  | 0.608526 | 0.800207 |
| `44ff79937f7775034771` | csbrain<br>tuev-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.545148 |  |  |  | 0.578070 | 0.770825 |
| `4870f7cbf35709d73a1b` | csbrain<br>tuev-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.692184 |  |  |  | 0.657492 | 0.840424 |
| `9eb5e1d4962e774501cf` | csbrain<br>tuev-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.602681 |  |  |  | 0.615067 | 0.796609 |
| `bfcf5c41adcec5208585` | csbrain<br>tuev-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.554445 |  |  |  | 0.557758 | 0.768792 |
| `ac16feff75ea1f02dc4d` | eegconformer<br>tuev-eegconformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.452355 |  |  |  | 0.413447 | 0.712976 |
| `720df50cd065134a53e9` | eegconformer<br>tuev-eegconformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.459587 |  |  |  | 0.470035 | 0.723421 |
| `816edc0a22f74877ebe2` | eegconformer<br>tuev-eegconformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.452522 |  |  |  | 0.380248 | 0.713306 |
| `c4d9a138d50e162b1613` | eegconformer<br>tuev-eegconformer_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.452355 |  |  |  | 0.413447 | 0.712976 |
| `c925f582ea14e5f39045` | eegconformer<br>tuev-eegconformer_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.481397 |  |  |  | 0.481068 | 0.731271 |
| `b9b7f5f0d37a9c1f8adb` | eegconformer<br>tuev-eegconformer_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.462375 |  |  |  | 0.434510 | 0.714401 |
| `bf788ef2f9ae0df37162` | eegconformer<br>tuev-eegconformer_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.491980 |  |  |  | 0.415627 | 0.730115 |
| `3ee173811d4778a73dad` | eegnet<br>tuev-eegnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.583591 |  |  |  | 0.480943 | 0.781142 |
| `14abf6d740515cd98db0` | eegnet<br>tuev-eegnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.422312 |  |  |  | 0.374394 | 0.715198 |
| `3e41f431d07bab0795fc` | eegnet<br>tuev-eegnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.430627 |  |  |  | 0.390385 | 0.720334 |
| `1c23fd56b44de714b8df` | eegnet<br>tuev-eegnet_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.583591 |  |  |  | 0.480943 | 0.781142 |
| `1be6208dadb3dacfcf04` | eegnet<br>tuev-eegnet_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.456952 |  |  |  | 0.322080 | 0.700000 |
| `4a710ac20bf017babfff` | eegnet<br>tuev-eegnet_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.407869 |  |  |  | 0.326553 | 0.703941 |
| `e1cacabb8e53e711bac2` | eegnet<br>tuev-eegnet_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.251160 |  |  |  | 0.255791 | 0.640094 |
| `9389c35d6b6d01861f3b` | eegpt<br>tuev-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.531132 |  |  |  | 0.527741 | 0.756596 |
| `b0ac30c1f6fa627b0118` | eegpt<br>tuev-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.609521 |  |  |  | 0.579757 | 0.796635 |
| `c6373a6bc023e51e7c5b` | eegpt<br>tuev-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.580261 |  |  |  | 0.431817 | 0.776443 |
| `18b7fe2073072c9fa5c2` | eegpt<br>tuev-eegpt_scratch_20260916 | scratch | 0 | completed_record<br>clinical_run | 0.556491 |  |  |  | 0.449187 | 0.765436 |
| `d1ce5b89da09b8857779` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.506949 |  |  |  | 0.415589 | 0.749407 |
| `d4ed6766b304d5b0a0cb` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.443155 |  |  |  | 0.382487 | 0.717436 |
| `c35193f2de6b2b95bc97` | ffcl<br>tuev-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.482729 |  |  |  | 0.514575 | 0.742877 |
| `c49316cb9796f1806023` | ffcl<br>tuev-ffcl_rebuild20261021 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.401800 |  |  |  | 0.561098 | 0.664510 |
| `594d5276b3cacf6fd9cd` | labram<br>tuev-labram_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run | 0.259273 |  |  |  | 0.251745 | 0.590666 |
| `a0bc388f9e10e532a5b8` | quad16前端 → biot<br>tuev-biot_quad16_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.555970 |  |  |  | 0.581911 | 0.764719 |
| `55fbfa88edd2064daa2a` | quad16前端 → cbramod<br>tuev-cbramod_quad16_20260915 | scratch | 0 | completed_record<br>clinical_run | 0.653318 |  |  |  | 0.626417 | 0.818190 |
| `c224bf70dfb8f34b0d3c` | quad_range前端 → biot<br>tuev-biot_quad_range_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.596107 |  |  |  | 0.517116 | 0.783315 |
| `a648caa6f8503d41cbfe` | quad_range前端 → biot<br>tuev-biot_range_identity_sean | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.399942 |  |  |  | 0.431714 | 0.693495 |
| `5bb5420d490dbe52ff1c` | quad_range前端 → biot<br>tuev-biot_range_identity_sean | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.385040 |  |  |  | 0.391819 | 0.682747 |
| `58cb1fab54a2e98b0364` | quad_range前端 → biot<br>tuev-biot_range_identity_sean | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.442964 |  |  |  | 0.417315 | 0.713248 |
| `e9622af3887a49819b8b` | quad_range前端 → biot<br>tuev-biot_range_measured_sean | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.582874 |  |  |  | 0.520182 | 0.782344 |
| `57b506accee5e966099b` | quad_range前端 → biot<br>tuev-biot_range_measured_sean | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.652658 |  |  |  | 0.552080 | 0.821159 |
| `ffabf1d980267c7e1ecd` | quad_range前端 → cbramod<br>tuev-cbramod_quad_range_20260916 | scratch | 0 | completed_record<br>clinical_run | 0.613269 |  |  |  | 0.520963 | 0.795867 |
| `e8622ac70ec988a0ab71` | quad_range前端 → cbramod<br>tuev-cbramod_range_measured_sean | scratch | 1 | completed_record<br>clinical_run | 0.537164 |  |  |  | 0.541155 | 0.760190 |
| `5941e653b0a08f5cb1c9` | quad_range前端 → cbramod<br>tuev-cbramod_range_measured_sean | scratch | 2 | completed_record<br>clinical_run | 0.610752 |  |  |  | 0.555865 | 0.794975 |
| `ac9c34c3b0ab585a0e34` | reve<br>tuev-reve_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.718846 |  |  |  | 0.650589 | 0.853642 |
| `6f62b5521e1795c3c9e1` | reve<br>tuev-reve_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.642289 |  |  |  | 0.628097 | 0.817192 |
| `2a7d891ead3271f07f22` | reve<br>tuev-reve_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.692703 |  |  |  | 0.664364 | 0.842427 |
| `f6c606bd1bd131552f3e` | reve<br>tuev-reve_scratch | scratch | 0 | completed_record<br>clinical_run | 0.303331 |  |  |  | 0.322602 | 0.657805 |
| `f21c3e1c35e1de247b31` | reve<br>tuev-reve_scratch | scratch | 1 | completed_record<br>clinical_run | 0.330657 |  |  |  | 0.383050 | 0.657079 |
| `40f3200955d83fb22e53` | reve<br>tuev-reve_scratch | scratch | 2 | completed_record<br>clinical_run | 0.322725 |  |  |  | 0.403949 | 0.649883 |
| `1d318f4cfe9f5dfc9669` | reve_paclockfe宿主转接<br>tuev-reve_crofremo | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.543685 |  |  |  | 0.523146 | 0.761218 |
| `9f3467f2d21650e2b0de` | reve_paclockfe宿主转接<br>tuev-reve_crofremo | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.596618 |  |  |  | 0.585872 | 0.788847 |
| `114eead9f94dcce7ae06` | reve_paclockfe宿主转接<br>tuev-reve_crofremo | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.528178 |  |  |  | 0.515349 | 0.755397 |
| `d7462c360dbc79a7d203` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.466508 |  |  |  | 0.469501 | 0.731937 |
| `2b16ed7d6a421308ce80` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.519888 |  |  |  | 0.497857 | 0.751400 |
| `91d047dbe25bdbc0d2cd` | sparcnet<br>tuev-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.479055 |  |  |  | 0.428989 | 0.722764 |
| `37507fef522866688e5d` | sparcnet<br>tuev-sparcnet_n20000 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.364227 |  |  |  | 0.386668 | 0.674806 |
| `63a25c49b533171e63d5` | sparcnet<br>tuev-sparcnet_n2160 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.491674 |  |  |  | 0.393034 | 0.730426 |
| `ff70d0bda94f552f4bcd` | sparcnet<br>tuev-sparcnet_n6720 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.553173 |  |  |  | 0.405231 | 0.755029 |
| `e84c1e3c448fd21eba8a` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.513858 |  |  |  | 0.489612 | 0.750815 |
| `9de1fe4bdc185627002b` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.491640 |  |  |  | 0.451040 | 0.734073 |
| `5aebd2eea3ccfac3b1f9` | st_transformer<br>tuev-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.496162 |  |  |  | 0.478596 | 0.738325 |
| `12fae1703121e0e0d84c` | tfm<br>tuev-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.654269 |  |  |  | 0.608754 | 0.818690 |
| `7949d539cfc16bdf8839` | tfm<br>tuev-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.675942 |  |  |  | 0.615497 | 0.830459 |
| `778c1bfbf66bbe297ac8` | tfm<br>tuev-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.663383 |  |  |  | 0.578310 | 0.819914 |
| `87ea2faa06353d553b26` | tfm<br>tuev-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.629946 |  |  |  | 0.555328 | 0.805873 |
| `5c7cbbea750029ddfe11` | tfm<br>tuev-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.651017 |  |  |  | 0.600768 | 0.815236 |
| `ec1d1469c085fe61e7cb` | tfm<br>tuev-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.671506 |  |  |  | 0.623712 | 0.826588 |
| `53e5cfa4832a1612db65` | tfm<br>tuev-tfm_pretoken_scratchenc_20260916 | scratch | 0 | completed_record<br>clinical_run | 0.581460 |  |  |  | 0.419527 | 0.779036 |
| `19a21d498318a626bdc5` | 旧CFC集成（须绑定原源码）<br>tuev-cfft_cfc_h2_twoaxis_20261029 | not_explicitly_recorded | 0 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `9e46a74f83a2d41d1fac` | 旧CFC集成（须绑定原源码）<br>tuev-cfft_cfc_h2_twoaxis_20261029 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.612626 |  |  |  | 0.641600 | 0.800896 |
| `b8980215baeb6bbde9f1` | 旧CFC集成（须绑定原源码）<br>tuev-cfft_cfc_residual_twoaxis_20261030 | not_explicitly_recorded | 0 | incomplete_status_unknown<br>training_progress |  |  |  |  |  |  |
| `01fe6bb7c072e1c35f17` | 显式PAC测量前端<br>tuev-cfft_explicit_pac_measured_20261028 | not_explicitly_recorded | 0 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `2039273e0e7ea0cf3592` | 显式PAC测量前端<br>tuev-cfft_explicit_pac_measured_notch_20261028 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.694206 |  |  |  | 0.618484 | 0.842272 |
| `d6a86922966695892b66` | 显式PAC测量前端 / 测量:identity<br>tuev-cfft_explicit_pac_identity_20261028 | not_explicitly_recorded | 0 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `1fbb709702284553518a` | 显式PAC测量前端 / 测量:identity<br>tuev-cfft_explicit_pac_identity_notch_20261028 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.683434 |  |  |  | 0.621070 | 0.835048 |
| `e822a1b839bf2afe5295` | 频带关系内容版<br>tuev-paclock_ft_base_partial | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.707111 |  |  |  | 0.571621 | 0.840675 |
| `74fc821af41e66363d4c` | 频带关系内容版<br>tuev-paclock_ft_base_v2_v2 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.643814 |  |  |  | 0.510644 | 0.807879 |
| `17c8b1d9fe1a11b5181a` | 频带关系内容版<br>tuev-paclock_pt_base | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.669075 |  |  |  | 0.587821 | 0.821897 |
| `e9d0651285c165da6ecb` | 频带关系内容版<br>tuev-paclock_pt_base | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.669842 |  |  |  | 0.616329 | 0.823936 |
| `4f82522c870a8ce6af53` | 频带关系内容版<br>tuev-paclock_pt_base | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.663420 |  |  |  | 0.608376 | 0.824626 |
| `7358a48b01ffb5c5da83` | 频带关系内容版<br>tuev-paclock_pt_base | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.641481 |  |  |  | 0.645906 | 0.809871 |
| `8e4a97763d5197ee58ca` | 频带关系内容版<br>tuev-paclock_pt_base | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.685882 |  |  |  | 0.633418 | 0.837367 |
| `f21ab0eafbb02fb7f211` | 频带关系内容版<br>tuev-paclock_pt_base | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.687449 |  |  |  | 0.646888 | 0.835421 |
| `0c10063a8fc4ef58faf7` | 频带关系内容版<br>tuev-paclock_pt_base_p200 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.666842 |  |  |  | 0.639936 | 0.825713 |
| `b5b2556a5a6aef873f5a` | 频带关系内容版<br>tuev-paclock_pt_base_p200 | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.683961 |  |  |  | 0.657257 | 0.834067 |
| `65e01440f44dbf3fb807` | 频带关系内容版<br>tuev-paclock_pt_base_p200 | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.687002 |  |  |  | 0.615367 | 0.837720 |
| `8fd16d61f721ddc9330f` | 频带关系内容版<br>tuev-paclock_pt_large | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.675852 |  |  |  | 0.618084 | 0.820094 |
| `fc990da48a20f89a74f5` | 频带关系内容版<br>tuev-paclock_pt_large | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.662805 |  |  |  | 0.698946 | 0.824384 |
| `47857f878a6b959f8e25` | 频带关系内容版<br>tuev-paclock_pt_large | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.618262 |  |  |  | 0.639688 | 0.800965 |
| `a100121a04fd8eac915b` | 频带关系内容版<br>bench-A_sdpa | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.638414 |  |  |  | 0.471225 | 0.805989 |
| `a3906dc99aa4d78bb97a` | 频带关系内容版<br>bench-B_bf16 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.636530 |  |  |  | 0.468505 | 0.804684 |
| `e617c28d8f983fb06753` | 频带关系内容版<br>bench-C_bf16_b128 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.475820 |  |  |  | 0.366219 | 0.726025 |
| `bef5692a4b87951dccc2` | 频带关系内容版<br>bench-D_bf16_b256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.364645 |  |  |  | 0.252996 | 0.657964 |
| `2b79fddca0a3baee4a51` | 频带关系内容版<br>bench-E_ram_b32 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.638414 |  |  |  | 0.471225 | 0.805989 |
| `d60ea654a8e01f661fe5` | 频带关系内容版<br>bench-F_ram_b128 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.477425 |  |  |  | 0.364674 | 0.726845 |
| `25ce93e553026fcd1c21` | 频带关系内容版<br>bench-G_ram_b128_bf16 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.475820 |  |  |  | 0.366219 | 0.726025 |
| `d2c640056a55e6786443` | 频带关系内容版<br>bench-H_ram_b256_bf16 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.364645 |  |  |  | 0.252996 | 0.657964 |
| `54a8181fb89f9e04ae5c` | 频带关系内容版<br>tuev-cand_aug | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.630397 |  |  |  | 0.478740 | 0.801890 |
| `ae0ea5973a05a6d7c47d` | 频带关系内容版<br>tuev-cand_aug_drop03 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.625833 |  |  |  | 0.468673 | 0.799138 |
| `f4493da119d0821821d8` | 频带关系内容版<br>tuev-cand_band_aug | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.582673 |  |  |  | 0.447510 | 0.780232 |
| `65709da77cc897dc11cd` | 频带关系内容版<br>tuev-cand_base | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.599915 |  |  |  | 0.468390 | 0.788825 |
| `9b82f183c701a6aeb29b` | 频带关系内容版<br>tuev-cand_base | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.643121 |  |  |  | 0.585572 | 0.814506 |
| `136162ff736dbcbe56f2` | 频带关系内容版<br>tuev-cand_base | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.644167 |  |  |  | 0.554164 | 0.813779 |
| `0cbf3128a20bba13ba46` | 频带关系内容版<br>tuev-cand_base | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.635632 |  |  |  | 0.512428 | 0.807376 |
| `b9bf2a50216c5fafce15` | 频带关系内容版<br>tuev-cand_dense | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.581921 |  |  |  | 0.425716 | 0.777624 |
| `6530ac74b9b2afe4b144` | 频带关系内容版<br>tuev-cand_dense_lr1e5 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.594716 |  |  |  | 0.443108 | 0.785805 |
| `b372c72de7168056a2d8` | 频带关系内容版<br>tuev-cand_dense_lr3e5 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.617305 |  |  |  | 0.495867 | 0.798737 |
| `bf2ec21936304fcf4cd8` | 频带关系内容版<br>tuev-cand_dense_patch100 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.688592 |  |  |  | 0.624384 | 0.837617 |
| `66f059e5ffa201f96482` | 频带关系内容版<br>tuev-cand_drop03 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.670869 |  |  |  | 0.600221 | 0.824849 |
| `fc7e6ff7141b3a98ece8` | 频带关系内容版<br>tuev-cand_e60_base | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.597019 |  |  |  | 0.456970 | 0.790023 |
| `8b65a3e53964ad41e890` | 频带关系内容版<br>tuev-cand_e60_patch100 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.707816 |  |  |  | 0.549449 | 0.839436 |
| `930acaa1e310942f457b` | 频带关系内容版<br>tuev-cand_e60_size_base | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.598649 |  |  |  | 0.652768 | 0.782700 |
| `5a3ef6139cc9a0187e6e` | 频带关系内容版<br>tuev-cand_e60_size_large | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.637994 |  |  |  | 0.552552 | 0.810065 |
| `dfe9fc02cf10b660a888` | 频带关系内容版<br>tuev-cand_head_attn | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.649923 |  |  |  | 0.621263 | 0.811570 |
| `d515314ca2de8d38f129` | 频带关系内容版<br>tuev-cand_head_band | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.585831 |  |  |  | 0.526770 | 0.783518 |
| `38aa8fe4e561f717369a` | 频带关系内容版<br>tuev-cand_heads8 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.641204 |  |  |  | 0.579930 | 0.809698 |
| `9dd90fc729a5e9f1b868` | 频带关系内容版<br>tuev-cand_ms | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.681600 |  |  |  | 0.629228 | 0.828897 |
| `124d169d6cde85318239` | 频带关系内容版<br>tuev-cand_ms | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.725994 |  |  |  | 0.594151 | 0.852276 |
| `7c03f1c271743959c199` | 频带关系内容版<br>tuev-cand_ms | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.668452 |  |  |  | 0.657311 | 0.821824 |
| `ecdd42f83b81b1854fb1` | 频带关系内容版<br>tuev-cand_p100_large | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.639260 |  |  |  | 0.544419 | 0.809185 |
| `71f8a4ec793d230efd9c` | 频带关系内容版<br>tuev-cand_p100_large | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.562180 |  |  |  | 0.597707 | 0.773861 |
| `520a5faa6660dc7d0ed2` | 频带关系内容版<br>tuev-cand_p100_large | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.643182 |  |  |  | 0.627040 | 0.817616 |
| `d8c2e679a9fb18650561` | 频带关系内容版<br>tuev-cand_p50_win100 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.609206 |  |  |  | 0.485364 | 0.797779 |
| `1e9541e17db0311888dd` | 频带关系内容版<br>tuev-cand_p50_win100 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.671518 |  |  |  | 0.535798 | 0.824486 |
| `df950c93356d98f56adf` | 频带关系内容版<br>tuev-cand_p50_win100 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.665860 |  |  |  | 0.570126 | 0.821053 |
| `09d29a7a3ffb825e8e25` | 频带关系内容版<br>tuev-cand_p50_win200 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.558215 |  |  |  | 0.470277 | 0.768805 |
| `cf3c21d1d50e3e33f8f6` | 频带关系内容版<br>tuev-cand_p50_win200 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.629893 |  |  |  | 0.499591 | 0.805018 |
| `0fa86205d87731f584f4` | 频带关系内容版<br>tuev-cand_p50_win200 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.643458 |  |  |  | 0.549885 | 0.809499 |
| `53543f3fb51f5c6ac397` | 频带关系内容版<br>tuev-cand_pacwin200 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.667751 |  |  |  | 0.514622 | 0.817901 |
| `b43127eade5d817455a9` | 频带关系内容版<br>tuev-cand_pacwin200 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.667348 |  |  |  | 0.515052 | 0.817777 |
| `955be022b59e598aa7e2` | 频带关系内容版<br>tuev-cand_pacwin200 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.616986 |  |  |  | 0.590853 | 0.801120 |
| `78285d583facf61bd35a` | 频带关系内容版<br>tuev-cand_pacwin200 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.615149 |  |  |  | 0.595713 | 0.798347 |
| `fedb27b784a3ba18fb18` | 频带关系内容版<br>tuev-cand_pacwin400 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.657838 |  |  |  | 0.588638 | 0.814009 |
| `787c92b35f0ee832aa3e` | 频带关系内容版<br>tuev-cand_patch100 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.707974 |  |  |  | 0.552500 | 0.839676 |
| `a764fed8a113f1da7967` | 频带关系内容版<br>tuev-cand_patch100 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.705561 |  |  |  | 0.547237 | 0.838318 |
| `9a45c487262428f614ff` | 频带关系内容版<br>tuev-cand_patch100 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.672841 |  |  |  | 0.588738 | 0.832051 |
| `c1ca8aff8e4fcf362fa6` | 频带关系内容版<br>tuev-cand_patch100 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.670108 |  |  |  | 0.528301 | 0.819589 |
| `38ea45227d7cd0d65775` | 频带关系内容版<br>tuev-cand_patch400 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.624882 |  |  |  | 0.594981 | 0.800585 |
| `cb3bf6da756bebc4dd00` | 频带关系内容版<br>tuev-cand_patch50 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.727764 |  |  |  | 0.562338 | 0.851897 |
| `b9296c0e7ac180fa9bb4` | 频带关系内容版<br>tuev-cand_patch50 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.723261 |  |  |  | 0.518724 | 0.849508 |
| `cb3cbf82942813ab5044` | 频带关系内容版<br>tuev-cand_patch50 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.671845 |  |  |  | 0.593744 | 0.826255 |
| `16310ed522f6ce1bcca1` | 频带关系内容版<br>tuev-cand_size_base | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.670717 |  |  |  | 0.617792 | 0.827595 |
| `c3417dfff73686d576a2` | 频带关系内容版<br>tuev-cand_size_large | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.617758 |  |  |  | 0.542588 | 0.797222 |
| `69b2c9f8fd68aa2d47be` | 频带关系内容版<br>tuev-crofremo_n1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.687331 |  |  |  | 0.666669 | 0.837993 |
| `9781a324949c03294e63` | 频带关系内容版<br>tuev-crofremo_n3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.708380 |  |  |  | 0.555939 | 0.841559 |
| `2aafdf16b41d4bccdc4b` | 频带关系内容版<br>tuev-crofremo_n4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.715315 |  |  |  | 0.703801 | 0.851594 |
| `852e520a9d4ee6314057` | 频带关系内容版<br>tuev-crofremo_n5 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.739109 |  |  |  | 0.692567 | 0.863185 |
| `29b50833038b24a539db` | 频带关系内容版<br>tuev-crofremo_s1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.707123 |  |  |  | 0.580691 | 0.843022 |
| `39676ecfedf8020bb784` | 频带关系内容版<br>tuev-crofremo_s3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.679604 |  |  |  | 0.640171 | 0.833922 |
| `1f9db42ae034b693d7e9` | 频带关系内容版<br>tuev-crofremo_s4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.728801 |  |  |  | 0.523716 | 0.848927 |
| `46f89c5a1dfd99108983` | 频带关系内容版<br>tuev-paclock_full | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.620557 |  |  |  | 0.463602 | 0.794869 |
| `f5b73ac99eee37c0309a` | 频带关系内容版<br>tuev-paclock_full | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.620659 |  |  |  | 0.600616 | 0.800784 |
| `318513f7e8d8e541a337` | 频带关系内容版<br>tuev-paclock_full | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.674092 |  |  |  | 0.557114 | 0.823494 |
| `89c99a3eaadf40ebff52` | 频带关系内容版<br>tuev-paclock_full | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.638612 |  |  |  | 0.550687 | 0.812827 |
| `a8efddda7cfc5518b918` | 频带关系内容版<br>tuev-paclock_full | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.622248 |  |  |  | 0.498312 | 0.802555 |
| `c03f882357d215e5f6bd` | 频带关系内容版<br>tuev-paclock_h12 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.718118 |  |  |  | 0.556513 | 0.847545 |
| `dcf3923a9b1e8a066059` | 频带关系内容版<br>tuev-paclock_h1stem | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.727764 |  |  |  | 0.562338 | 0.851897 |
| `01f0bc3aaabb760fd734` | 频带关系内容版<br>tuev-paclock_h2mont | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.718118 |  |  |  | 0.556513 | 0.847545 |
| `4e458ad2360ca8c1a640` | 频带关系内容版<br>tuev-paclock_nb16 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.722303 |  |  |  | 0.657915 | 0.852316 |
| `66b4815239f9849e4bfb` | 频带关系内容版<br>tuev-paclock_pac | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.622625 |  |  |  | 0.519769 | 0.801433 |
| `a1d7a1bde661a871178c` | 频带关系内容版<br>tuev-paclock_pac | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.635356 |  |  |  | 0.505327 | 0.808425 |
| `4c18fa45a0456bbe3b86` | 频带关系内容版<br>tuev-paclock_pac | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.619197 |  |  |  | 0.643481 | 0.797390 |
| `21aa9ca9bb9f835fabf7` | 频带关系内容版<br>tuev-paclock_pilot_frozen | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.620659 |  |  |  | 0.600616 | 0.800784 |
| `d4adfddc6b6d308170d3` | 频带关系内容版<br>tuev-paclock_pilot_frozen | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.638612 |  |  |  | 0.550687 | 0.812827 |
| `37fef19262834826ba57` | 频带关系内容版<br>tuev-paclock_pilot_frozen | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.622248 |  |  |  | 0.498312 | 0.802555 |
| `e26299f5e728d92d421d` | 频带关系内容版<br>tuev-paclock_pilot_unfiltered | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.648894 |  |  |  | 0.596230 | 0.815339 |
| `4a61ffbd62b260b75b33` | 频带关系内容版<br>tuev-paclock_pilot_unfiltered | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.562815 |  |  |  | 0.551479 | 0.772650 |
| `4dfa96086664b0473b1c` | 频带关系内容版<br>tuev-paclock_pilot_unfiltered | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.650969 |  |  |  | 0.482883 | 0.813045 |
| `b3e610273dbf097beea6` | 频带关系内容版<br>tuev-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.735155 |  |  |  | 0.567794 | 0.856248 |
| `0882a20bf20d287bbed4` | 频带关系内容版<br>tuev-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.747546 |  |  |  | 0.543585 | 0.859424 |
| `0bbf74a687fbd3bb30bc` | 频带关系内容版<br>tuev-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.715623 |  |  |  | 0.582507 | 0.844901 |
| `43a784553d069c2a44f7` | 频带关系内容版<br>tuev-paclock_rot_ms | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.683244 |  |  |  | 0.687678 | 0.828251 |
| `d3357c9b58f3b0f2f5c8` | 频带关系内容版<br>tuev-paclock_sphead | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.651958 |  |  |  | 0.589327 | 0.818452 |
| `b1d68f9d4b9672c9f3e6` | 频带关系内容版<br>tuev-paclock_v2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.727764 |  |  |  | 0.562338 | 0.851897 |
| `89e90bdd81e006ddc11f` | 频带关系内容版<br>tuev-paclock_v2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.723261 |  |  |  | 0.518724 | 0.849508 |
| `f7e27aeeb2f7ac0d99e0` | 频带关系内容版<br>tuev-paclock_v2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.671845 |  |  |  | 0.593744 | 0.826255 |
| `9f5b4e73d7500dc8783f` | 频带关系内容版<br>tuev-paclock_v2_n20000 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.676682 |  |  |  | 0.555244 | 0.821228 |
| `67425adaef03863ac7e2` | 频带关系内容版<br>tuev-paclock_v2_n2160 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.652331 |  |  |  | 0.464462 | 0.811647 |
| `10c43377ebea6b61379e` | 频带关系内容版<br>tuev-paclock_v2_n6720 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.709836 |  |  |  | 0.553231 | 0.842437 |
| `5467d201ff80167b3b4d` | 频带关系内容版<br>tuev-rot2_residual_control | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.722233 |  |  |  | 0.558954 | 0.850563 |
| `dbcab3cb34fc4f001360` | 频带关系内容版 / anchored<br>tuev-nb16_anchored_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.606832 |  |  |  | 0.448491 | 0.790541 |
| `a660007e429f350fb188` | 频带关系内容版 / carrier<br>tuev-nb16_carrier_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.721377 |  |  |  | 0.606356 | 0.848272 |
| `1e722336d108eb75d620` | 频带关系内容版 / local-residual<br>tuev-nb16_residual_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.649070 |  |  |  | 0.500835 | 0.809139 |
| `3883a6c883b9343744ee` | 频带关系内容版 / local-residual<br>tuev-rot2_residual_band | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.728280 |  |  |  | 0.572201 | 0.853038 |
| `4fe116a80ed25a1b5ee6` | 频带波形原始版<br>tuev-cf2_v1d192raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.572992 |  |  |  | 0.478554 | 0.775386 |
| `10ef979b61e42d7623f9` | 频带波形原始版<br>tuev-cf2_v1d192raw | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.535580 |  |  |  | 0.527029 | 0.756981 |
| `4b1430a2e43abd31b030` | 频带波形原始版<br>tuev-cf2_v1d256raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.600883 |  |  |  | 0.565988 | 0.791751 |
| `1d36c7cf0cb695544ff8` | 频带波形原始版<br>tuev-cf2_v1d256raw | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.580885 |  |  |  | 0.602964 | 0.777894 |
| `f2cf9687812c5173aa7b` | 频带波形原始版<br>tuev-cf2_v1d256raw | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.519362 |  |  |  | 0.478493 | 0.739009 |
| `6dbcb8f70ce3c0f8283a` | 频带波形原始版<br>tuev-cf2_v1raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.545566 |  |  |  | 0.479202 | 0.764187 |
| `573e28a31a7bc4c869c4` | 频带波形原始版<br>tuev-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.577303 |  |  |  | 0.566142 | 0.769989 |
| `138f61e90255ce6eb2fb` | 频带波形原始版<br>tuev-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.536032 |  |  |  | 0.487181 | 0.753904 |
| `5d32efe294dedb7739ef` | 频带波形原始版<br>tuev-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.494489 |  |  |  | 0.601935 | 0.732853 |

## shared16_count_not_recorded（2 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `f48f767a23f2b3e80c23` | labram<br>tuev-labram-shared16-pretrained-s0 | pretrained | 0 | completed_record<br>clinical_run | 0.602003 |  |  |  | 0.654664 | 0.796599 |
| `e3200830185721d7fd2b` | labram<br>tuev-labram-shared16-scratch-s0 | scratch | 0 | completed_record<br>clinical_run | 0.568542 |  |  |  | 0.379810 | 0.765613 |
