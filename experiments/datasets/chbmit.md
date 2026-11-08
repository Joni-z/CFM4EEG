# CHB-MIT 实验记录

[返回总入口](../README.md)

共 233 条记录。[完整精度及来源 CSV](chbmit.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## legacy_enriched_17386（210 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `0959fe660e80b8276f0e` | CfFT其他前端<br>chbmit-factorized_f0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.475054 |  | 0.896900 | 0.505435 |  |
| `8c49456acbab84847c02` | CfFT其他前端<br>chbmit-factorized_f1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.705042 |  | 0.935068 | 0.543478 |  |
| `020ffab060b0c4e986fb` | CfFT其他前端<br>chbmit-factorized_f2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.491689 |  | 0.898552 | 0.536144 |  |
| `0efa4b843b4435d8b751` | CfFT其他前端<br>chbmit-paclock_fg_stem | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.720988 |  | 0.918665 | 0.509058 |  |
| `d0947f7a66dd14d1a8fe` | CfFT其他前端<br>chbmit-paclock_fuse | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.719393 |  | 0.918021 | 0.641129 |  |
| `bd09dc601cd14c87d72b` | CfFT其他前端<br>chbmit-paclock_fusegate | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.744148 |  | 0.931805 | 0.648375 |  |
| `e9fa809853630e75f6f0` | CfFT其他前端<br>chbmit-paclock_fusegate | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.613819 |  | 0.903104 | 0.519898 |  |
| `94dd89cf49928feaf06b` | CfFT其他前端<br>chbmit-paclock_fusegate | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.598457 |  | 0.905104 | 0.567029 |  |
| `930cb3ddcfaff0e60a23` | CfFT其他前端<br>chbmit-paclock_fusegate_ms | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.285832 |  | 0.867775 | 0.555780 |  |
| `8e7090968803b61ed93c` | CfFT其他前端<br>chbmit-paclock_hybrid | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.747287 |  | 0.936793 | 0.528986 |  |
| `9c7f289466858795deeb` | CfFT其他前端<br>chbmit-paclock_hybrid_attn | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.720035 |  | 0.940317 | 0.567000 |  |
| `65f6c53c9f535d0623c3` | CfFT其他前端<br>chbmit-paclock_hybrid_gate | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.751265 |  | 0.939052 | 0.539826 |  |
| `283d764459b70261b404` | CfFT其他前端<br>chbmit-paclock_hybrid_gate | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.674453 |  | 0.907516 | 0.567000 |  |
| `41f519f1c093141b4cae` | CfFT其他前端<br>chbmit-paclock_hybrid_gate | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.721863 |  | 0.917221 | 0.581463 |  |
| `fcd5b5d3011e3045cd66` | Duplex<br>chbmit-cf2_v0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.569267 |  | 0.902148 | 0.536232 |  |
| `dd3dc276ab0e093d3a88` | Duplex<br>chbmit-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.630183 |  | 0.901898 | 0.500000 |  |
| `94247f377eaa9ad3223f` | Duplex<br>chbmit-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.513268 |  | 0.893147 | 0.536173 |  |
| `78f259525245214ab4b2` | Duplex<br>chbmit-cf2_v1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.646642 |  | 0.913857 | 0.516304 |  |
| `e8566a8b6b6ca073c2e4` | Duplex<br>chbmit-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.607690 |  | 0.912010 | 0.539855 |  |
| `6e27d75a6d5709d5e340` | Duplex<br>chbmit-cf2_v1d256 | not_explicitly_recorded | 1 | budget_truncated<br>clinical_run |  | 0.618252 |  | 0.908724 | 0.559695 |  |
| `61e6f5a2e0c057ac0ef4` | Duplex<br>chbmit-cf2_v1d256 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.606804 |  | 0.907275 | 0.572435 |  |
| `5aece30171a5c0da3b30` | Duplex<br>chbmit-cf2_v2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.628360 |  | 0.912054 | 0.539855 |  |
| `7683b14be54107e81f25` | Duplex-A128<br>chbmit-paclock_duplex_p200ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.544829 |  | 0.881940 | 0.586927 |  |
| `9f44d000284d835940d6` | Duplex-A128<br>chbmit-paclock_duplex_pt | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.663530 |  | 0.914487 | 0.538043 |  |
| `69a2efa7ab5f08b5569c` | Duplex-A128<br>chbmit-paclock_duplex_pt2 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.713402 |  | 0.929361 | 0.547101 |  |
| `3a285baa861b75f9b7a2` | Duplex-A128<br>chbmit-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.590720 |  | 0.935663 | 0.604956 |  |
| `e4d6f1d76d59da8e50e7` | Duplex-A128<br>chbmit-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.699140 |  | 0.910483 | 0.716895 |  |
| `d855cb9430fb88cf66f3` | Duplex-A128<br>chbmit-paclock_probe_v1 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.063092 |  | 0.749075 | 0.500000 |  |
| `5bff4811446cfec29c15` | Duplex-A128<br>chbmit-paclock_probe_v2 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.049908 |  | 0.691667 | 0.500000 |  |
| `9d98e4e00a6a24a33471` | Duplex-A128<br>chbmit-crofremo_s2_mi2101x_pilot | not_explicitly_recorded | 0 | budget_truncated<br>clinical_run |  | 0.720635 |  | 0.926595 | 0.519928 |  |
| `82a4d0d9d8d2ce9a04cb` | Duplex-A128<br>chbmit-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.712998 |  | 0.925644 | 0.516304 |  |
| `5856ce3535c0da8adcab` | Duplex-A128<br>chbmit-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.637748 |  | 0.911108 | 0.567000 |  |
| `b1c8590c0c5bbbec9308` | Duplex-A128<br>chbmit-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.744663 |  | 0.916604 | 0.559753 |  |
| `07ea10ae1f461baba8c9` | Duplex-A128<br>chbmit-paclock_duplex_ms | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.192746 |  | 0.823731 | 0.548416 |  |
| `52ff4562ef43cd75f8c1` | Duplex-A128<br>chbmit-paclock_duplex_p200 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.548640 |  | 0.883820 | 0.566971 |  |
| `1cf34ca6668ffcbda8cd` | Duplex-A128<br>chbmit-paclock_probe_rand | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.014737 |  | 0.472452 | 0.500000 |  |
| `98bbbdac6a7e789ddbea` | Duplex-F192<br>chbmit-cf2_v1d192_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.627034 |  | 0.913589 | 0.612173 |  |
| `9c4f537ac15f958fa511` | Duplex-F192<br>chbmit-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.716347 |  | 0.928786 | 0.530768 |  |
| `664c566eead96969cdd9` | Duplex-F192<br>chbmit-cf2_v1d192 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.651011 |  | 0.913741 | 0.550695 |  |
| `491083e7b68457392add` | Duplex-F192<br>chbmit-cf2_v1d192 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.639964 |  | 0.888478 | 0.539855 |  |
| `86755a84c70070084260` | Duplex-F192<br>chbmit-cf2_v3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.670885 |  | 0.904878 | 0.538043 |  |
| `0b5b57c729d779e1d828` | Duplex-F192 / own<br>chbmit-cf2_v1d192own | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.651313 |  | 0.916851 | 0.570623 |  |
| `7042f91b210d306a0bb3` | Duplex-F192 / own<br>chbmit-cf2_v1d192own | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.584947 |  | 0.911954 | 0.548855 |  |
| `bf106fe1e8b2935e6e10` | Duplex-F192 / own<br>chbmit-cf2_v1d192own | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.673829 |  | 0.912078 | 0.579652 |  |
| `1b3a3d35b6fa44201532` | Quad-Range<br>chbmit-quad_range_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.696913 |  | 0.912909 | 0.608550 |  |
| `e66e05156781b5d57edc` | Quad-Range<br>chbmit-quad_range_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.709807 |  | 0.921328 | 0.586840 |  |
| `9d938a597a8130f674fe` | Quad-Range<br>chbmit-quad_range_Z1s_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.727366 |  | 0.907095 | 0.617608 |  |
| `6a19b026c49c0413bccb` | Quad-Range<br>chbmit-quad_range_Z5s_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.617915 |  | 0.891198 | 0.586840 |  |
| `48fea7d9a2325051cf61` | Quad-Range<br>chbmit-range_twoaxis_measured_20260917 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.604655 |  | 0.912784 | 0.552449 |  |
| `b9cffac1c80c09aa3090` | Quad-Range<br>chbmit-range_twoaxis_measured_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.623756 |  | 0.896857 | 0.630230 |  |
| `31e3bcecae3fd2778b94` | Quad-Range / envelope-only<br>chbmit-quad_range_envelope_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.543221 |  | 0.900020 | 0.588680 |  |
| `4b11e341c0f71ff8bc71` | Quad-Range / envelope-only<br>chbmit-range_twoaxis_envelope_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.561782 |  | 0.906614 | 0.590492 |  |
| `25e0261048bd218f4b63` | Quad-Range / identity<br>chbmit-legacy-triaxial-identity-20261105 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.699413 | 0.701707 | 0.917884 | 0.630318 |  |
| `45fcaf9777b6c421e7b1` | Quad-Range / identity<br>chbmit-legacy-triaxial-identity-20261105 | not_explicitly_recorded | 2 | interim_eval<br>interim_checkpoint_eval |  | 0.699413 | 0.701707 | 0.917884 | 0.630318 |  |
| `3d740a02a7dd791d413e` | Quad-Range / identity<br>chbmit-range_identity_sean | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.671299 |  | 0.931230 | 0.615796 |  |
| `3bcd7d9ff0df48467fed` | Quad-Range / identity<br>chbmit-range_twoaxis_identity_20260917 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.664544 |  | 0.907189 | 0.588651 |  |
| `ac965137018aef2d5c33` | Quad-Range / identity<br>chbmit-range_twoaxis_identity_20260917 | not_explicitly_recorded | 2 | budget_truncated<br>clinical_run |  | 0.650387 |  | 0.929242 | 0.597738 |  |
| `24fe2f9e933eb375ccab` | Quad-Range / identity, content:envelope<br>chbmit-demodulation-envelope-20261106 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.743611 | 0.744869 | 0.924440 | 0.615796 |  |
| `048b390ed53b7fd20955` | Quad-Range / magnitude<br>chbmit-quad_range_weight_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.645345 |  | 0.926163 | 0.635723 |  |
| `302ef4e86ef47632d55a` | Quad-Range / own<br>chbmit-quad_range_own_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.468019 |  | 0.860343 | 0.500000 |  |
| `5c06d8b88be1bbc52455` | Quad-Range / own<br>chbmit-range_twoaxis_own_20260917 | not_explicitly_recorded | 2 | budget_truncated<br>clinical_run |  | 0.507140 |  | 0.882413 | 0.644781 |  |
| `19f5603799e0f11ac4cc` | Quad16<br>chbmit-nb16_quadrature_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.731917 |  | 0.924594 | 0.561536 |  |
| `35354beab97a4c89be70` | WavePatch 学习内容行三轴<br>chbmit-wave_patch_threeaxis-20261105 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.641878 | 0.642482 | 0.927718 | 0.538014 |  |
| `c2db3edf9d90a15322ba` | biot<br>chbmit-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.471274 |  | 0.871741 | 0.668975 |  |
| `9be06ecbbd41bbc9549d` | biot<br>chbmit-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.599036 |  | 0.878047 | 0.583100 |  |
| `76b5d8d0930526d86cf7` | biot<br>chbmit-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.609992 |  | 0.871751 | 0.604780 |  |
| `56e2b891b4a2e6e7b6b2` | biot<br>chbmit-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.649124 |  | 0.895344 | 0.705967 |  |
| `d6e23ee6e55ac4a8afed` | biot<br>chbmit-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.518029 |  | 0.859457 | 0.586606 |  |
| `ae5c9033f7c97e957fef` | biot<br>chbmit-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.531273 |  | 0.869473 | 0.579389 |  |
| `5bf4658a839954227179` | biot_tfm_host宿主转接<br>chbmit-biot_tfm_host_20260918 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.617723 |  | 0.901289 | 0.604985 |  |
| `7f1e46e9abcb2586b633` | cbramod<br>chbmit-cbramod_pac | pretrained | 0 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `524909c3a0f302ad3964` | cbramod<br>chbmit-cbramod_pac | pretrained | 1 | completed_record<br>clinical_run |  | 0.403624 |  | 0.859209 | 0.559549 |  |
| `465abfc35d09743c115a` | cbramod<br>chbmit-cbramod_pac | pretrained | 2 | completed_record<br>clinical_run |  | 0.477367 |  | 0.927817 | 0.543391 |  |
| `1dea48861050d07256c8` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.120281 |  | 0.730403 | 0.528927 |  |
| `cc402172ab395ff5dfbe` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.268739 |  | 0.828709 | 0.501753 |  |
| `efbbaabccd8e5a9862b7` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.257469 |  | 0.794516 | 0.554319 |  |
| `0d802664b4f1bb579157` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.290915 |  | 0.822592 | 0.537985 |  |
| `70ae8670efee0ba06ef0` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `986e2d03f8653b8316f1` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.013966 |  | 0.480453 | 0.500000 |  |
| `1ddf577f626c3c746f53` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.431795 |  | 0.869767 | 0.640691 |  |
| `4a777de86a554937c644` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.288269 |  | 0.795450 | 0.566883 |  |
| `cac8e5c15ecd80044602` | cbramod<br>chbmit-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.258448 |  | 0.811777 | 0.500000 |  |
| `fbde535abe811b93b850` | cbramod<br>chbmit-cbramod_ptn_ds4 | scratch | 0 | completed_record<br>clinical_run |  | 0.256385 |  | 0.493618 | 0.500000 |  |
| `f06b50170bf7244e3526` | cbramod<br>chbmit-cbramod_ptn_tuegc | scratch | 0 | completed_record<br>clinical_run |  | 0.439551 |  | 0.554947 | 0.500000 |  |
| `1072e7d3642b4b377616` | cbramod<br>chbmit-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `330abf5b6ea0ce91255b` | cbramod<br>chbmit-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.414800 |  | 0.832924 | 0.648258 |  |
| `3ec1277c31fd1af95724` | cbramod<br>chbmit-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `9a505a67dd40ccbf7dce` | cbramod<br>chbmit-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.015904 |  | 0.510749 | 0.500000 |  |
| `e2e3ec8ee36f50285faa` | cbramod<br>chbmit-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `7575f22c4dfab6ad4d32` | cbramod<br>chbmit-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `7dfaadeea3c0e19cc03e` | cbramod<br>chbmit-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.454587 |  | 0.879126 | 0.657229 |  |
| `7ffffe22fd03d57629e7` | cbramod<br>chbmit-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `94ae81fc0e3c9b1b5176` | cbramod<br>chbmit-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.475548 |  | 0.481405 | 0.500000 |  |
| `bfa2072abc419d8cb254` | cbramod<br>chbmit-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.033062 |  | 0.635401 | 0.500000 |  |
| `1b3d45da02c897b901c3` | cbramod<br>chbmit-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.081258 |  | 0.628652 | 0.500000 |  |
| `448d3036860c7fe2bce3` | cbramod<br>chbmit-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.513286 |  | 0.505435 | 0.500000 |  |
| `73dbb050d7b7ac000002` | cbramod<br>chbmit-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.507938 |  | 0.500029 | 0.500000 |  |
| `a54472b7f3913fdb8766` | cbramod<br>chbmit-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.056774 |  | 0.598611 | 0.500000 |  |
| `e1118fd5ec00a50d5ae3` | cbramod<br>chbmit-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.007956 |  | 0.497515 | 0.500000 |  |
| `21a0278531ff93bbf6ea` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_add_cpl | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.465006 |  | 0.857926 | 0.659128 |  |
| `4f5a405694294f2bc2bd` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_add_cpl | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.441397 |  | 0.853065 | 0.646418 |  |
| `976ca03f1ce8a58863fa` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_add_cpl | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.314076 |  | 0.813068 | 0.581317 |  |
| `838ca07158185527d08c` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_add_raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.340648 |  | 0.804616 | 0.615124 |  |
| `49fd99aec5a4733fd772` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_add_raw | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.513672 |  | 0.912636 | 0.655417 |  |
| `cb7bf39aaf99109d0e6c` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_add_raw | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.450452 |  | 0.888614 | 0.654979 |  |
| `5c7c3b5b487aef944742` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_pacfe_pac | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `c1b134c158ca2f9ec475` | cbramod_paclockfe宿主转接<br>chbmit-cbramod_pacfe_raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.507938 |  | 0.500029 | 0.500000 |  |
| `81710376f15c56a0a9cd` | cnn_transformer<br>chbmit-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.543943 |  | 0.916293 | 0.709181 |  |
| `858dbddaebfd96d35a8a` | cnn_transformer<br>chbmit-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.552811 |  | 0.914381 | 0.604868 |  |
| `79c470de50b6f580e209` | cnn_transformer<br>chbmit-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.420209 |  | 0.901019 | 0.612056 |  |
| `a6ca7203c0f9c2c3243a` | contrawr<br>chbmit-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.532695 |  | 0.907536 | 0.640924 |  |
| `169e49c3bbf429ed3a3b` | contrawr<br>chbmit-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.522924 |  | 0.900747 | 0.650099 |  |
| `3600dc2f2c3079bfa3b9` | contrawr<br>chbmit-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.466056 |  | 0.863127 | 0.617491 |  |
| `c9faddc161bbfe9e0d7c` | csbrain<br>chbmit-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.402010 |  | 0.885658 | 0.525362 |  |
| `79959e533200558dd7fb` | csbrain<br>chbmit-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.422878 |  | 0.890363 | 0.568694 |  |
| `e96315c3e4db55f4b5dd` | csbrain<br>chbmit-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.247448 |  | 0.780319 | 0.543069 |  |
| `1eca834c247104c911eb` | eegconformer<br>chbmit-eegconformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.386475 |  | 0.791375 | 0.500000 |  |
| `fa2567f605d9d77ab3de` | eegconformer<br>chbmit-eegconformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.021798 |  | 0.500787 | 0.500000 |  |
| `9579c6fd9531acf1d24d` | eegconformer<br>chbmit-eegconformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.507937 |  | 0.500000 | 0.500000 |  |
| `e2bead9ed7994cc82b78` | eegconformer<br>chbmit-eegconformer_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.275202 |  | 0.671457 | 0.530797 |  |
| `45cc5317ecaddba4c95a` | eegconformer<br>chbmit-eegconformer_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.386475 |  | 0.791375 | 0.500000 |  |
| `3a5b04d509a4c0f663a5` | eegconformer<br>chbmit-eegconformer_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.346231 |  | 0.738870 | 0.528927 |  |
| `4d1f051e4816433ed34d` | eegconformer<br>chbmit-eegconformer_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.352605 |  | 0.751177 | 0.604546 |  |
| `823a15e3bee70a0ff172` | eegnet<br>chbmit-eegnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.101126 |  | 0.713400 | 0.536699 |  |
| `806ab4fa5b52cc5edcb2` | eegnet<br>chbmit-eegnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.030168 |  | 0.652822 | 0.499883 |  |
| `0bc9065a88a16fffa12d` | eegnet<br>chbmit-eegnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.290412 |  | 0.866298 | 0.768287 |  |
| `caf9f974649018e39c6e` | eegnet<br>chbmit-eegnet_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.073964 |  | 0.719626 | 0.500000 |  |
| `dc3efd1875211b1b12bc` | eegnet<br>chbmit-eegnet_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.098655 |  | 0.747591 | 0.582075 |  |
| `111504c26a49bd0a4af9` | eegnet<br>chbmit-eegnet_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.114726 |  | 0.727377 | 0.505435 |  |
| `bb8fea37cad5dfdaa830` | eegnet<br>chbmit-eegnet_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.101126 |  | 0.713400 | 0.536699 |  |
| `c88be091c71c1c0cb82d` | eegpt<br>chbmit-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.404070 |  | 0.899372 | 0.521681 |  |
| `ef18fbf5d35ac545a77d` | eegpt<br>chbmit-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.298560 |  | 0.846587 | 0.618688 |  |
| `e86b36e1e0a6eb761207` | eegpt<br>chbmit-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.140739 |  | 0.826211 | 0.500000 |  |
| `83a3b2cf364ede6efe71` | ffcl<br>chbmit-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.577355 |  | 0.924136 | 0.613838 |  |
| `ddc26663d53cbfe5c491` | ffcl<br>chbmit-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.473550 |  | 0.903217 | 0.631983 |  |
| `651beffef8f3723f55de` | ffcl<br>chbmit-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.551292 |  | 0.909829 | 0.603086 |  |
| `8c6d8927539347b8a925` | labram<br>chbmit-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.542523 |  | 0.938497 | 0.639054 |  |
| `35499fe902b9d5f50d84` | labram<br>chbmit-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.349701 |  | 0.857951 | 0.548855 |  |
| `c9a633aacda5f295efaa` | labram<br>chbmit-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.463436 |  | 0.816295 | 0.568694 |  |
| `5b03943b9f7a087459f1` | labram<br>chbmit-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.402035 |  | 0.841351 | 0.649719 |  |
| `ead48a266be91775846f` | labram<br>chbmit-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.337418 |  | 0.839098 | 0.592187 |  |
| `4ceb5e212a26edd407be` | labram<br>chbmit-labram_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.445871 |  | 0.877938 | 0.565071 |  |
| `500f42de1ad3572ae371` | labram<br>chbmit-labram_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.371942 |  | 0.838106 | 0.519869 |  |
| `79217793a3b24a9884a9` | labram<br>chbmit-labram_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.304743 |  | 0.788226 | 0.586547 |  |
| `b4fd34d92dab15cf154e` | labram<br>chbmit-labram_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.429075 |  | 0.868235 | 0.559666 |  |
| `fc57e525000b0e20f75c` | labram_paclockfe宿主转接<br>chbmit-labram_crofremo | scratch | 0 | budget_truncated<br>clinical_run |  | 0.377148 |  | 0.845264 | 0.635548 |  |
| `a100886e129c69abfb80` | labram_paclockfe宿主转接<br>chbmit-labram_crofremo | scratch | 1 | budget_truncated<br>clinical_run |  | 0.318862 |  | 0.837712 | 0.583129 |  |
| `d14f66d945cdf6837d7c` | labram_paclockfe宿主转接<br>chbmit-labram_crofremo | scratch | 2 | budget_truncated<br>clinical_run |  | 0.437793 |  | 0.881408 | 0.608520 |  |
| `55aae66f3b964f004f05` | quad16前端 → biot<br>chbmit-biot_quad16_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.545498 |  | 0.877198 | 0.541550 |  |
| `2f016fd4f2e244e750b1` | quad_range前端 → biot<br>chbmit-biot_quad_range_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.473792 |  | 0.818095 | 0.661027 |  |
| `66c32a65ddca3048f00e` | quad_range前端 → biot<br>chbmit-biot_range_identity_sean | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.685205 |  | 0.916215 | 0.570535 |  |
| `2c4bd33bfe0a3ade4a7f` | quad_range前端 → biot<br>chbmit-biot_range_identity_sean | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.646232 |  | 0.927708 | 0.650187 |  |
| `c5863fdfc1595e699e14` | quad_range前端 → biot<br>chbmit-biot_range_identity_sean | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.554469 |  | 0.901955 | 0.628506 |  |
| `10d38891541ca6fbce55` | quad_range前端 → biot<br>chbmit-biot_range_measured_sean | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.453602 |  | 0.811818 | 0.641158 |  |
| `df4ce8cdfbbf29af2308` | quad_range前端 → biot<br>chbmit-biot_range_measured_sean | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.465113 |  | 0.834477 | 0.661027 |  |
| `8c6d6e24496341fd0a74` | quad_range前端 → cbramod<br>chbmit-cbramod_range_measured_sean | scratch | 0 | completed_record<br>clinical_run |  | 0.273786 |  | 0.715117 | 0.583187 |  |
| `36539a28d8ab3463c24a` | reve<br>chbmit-reve_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.725837 |  | 0.936851 | 0.704330 |  |
| `a97e24f5dc8f70371bf7` | reve<br>chbmit-reve_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.357881 |  | 0.823687 | 0.570594 |  |
| `0909d5ab32c23a9f8293` | reve<br>chbmit-reve_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.657013 |  | 0.932486 | 0.621201 |  |
| `f713cb9d18c5b2f12ff0` | sparcnet<br>chbmit-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.233745 |  | 0.838697 | 0.509029 |  |
| `f3a82922f7b5c295df47` | sparcnet<br>chbmit-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.115263 |  | 0.764221 | 0.507246 |  |
| `c80c3d3db12f8a0331f1` | sparcnet<br>chbmit-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.243826 |  | 0.795433 | 0.521710 |  |
| `404796f6dce7f6f618be` | st_transformer<br>chbmit-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.309926 |  | 0.833656 | 0.673268 |  |
| `7ef07345d9ee363b5b8a` | st_transformer<br>chbmit-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.109795 |  | 0.629594 | 0.477480 |  |
| `f32c3f8958633814433a` | st_transformer<br>chbmit-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.302363 |  | 0.699257 | 0.514592 |  |
| `4ac42db07020d0306b60` | tfm<br>chbmit-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.636585 |  | 0.923108 | 0.617578 |  |
| `69ea96cad40e1c4e26fd` | tfm<br>chbmit-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.645655 |  | 0.930249 | 0.650129 |  |
| `a88d8e1ae9008dfae76c` | tfm<br>chbmit-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.598404 |  | 0.933965 | 0.635636 |  |
| `9313390815cc919aca52` | 模型未记录<br>chbmit-c-eval-legacy-20261104 | not_explicitly_recorded |  | interim_eval<br>interim_checkpoint_eval |  | 0.534567 | 0.535668 | 0.882914 | 0.628506 |  |
| `3274bb6a4e9e76eeeb14` | 频带关系内容版<br>chbmit-paclock_p200_pt | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.460681 |  | 0.894510 | 0.579447 |  |
| `030b87206a4c9c292ade` | 频带关系内容版<br>chbmit-paclock_p200_pt_notok | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.527691 |  | 0.896512 | 0.565159 |  |
| `84bd1941d8a2848c7976` | 频带关系内容版<br>chbmit-paclock_pt_base | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.705799 |  | 0.940522 | 0.641100 |  |
| `f14dca57bf723305316b` | 频带关系内容版<br>chbmit-paclock_pt_base | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.562570 |  | 0.908826 | 0.500000 |  |
| `7a914fb9d9ee9bfd1d5b` | 频带关系内容版<br>chbmit-paclock_pt_base | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.660210 |  | 0.919312 | 0.588680 |  |
| `3671f89ea2868ae4d611` | 频带关系内容版<br>chbmit-paclock_pt_base | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.683133 |  | 0.942727 | 0.659216 |  |
| `467b0aec305b53c3d679` | 频带关系内容版<br>chbmit-paclock_pt_excl | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.641009 |  | 0.944041 | 0.682591 |  |
| `0cf64356e0a0bb6ab5c4` | 频带关系内容版<br>chbmit-paclock_pt_large | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.563393 |  | 0.909806 | 0.514493 |  |
| `095f6edc92723cc9a5ab` | 频带关系内容版<br>chbmit-paclock_pt_large | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.632871 |  | 0.943303 | 0.500000 |  |
| `5617bf8729e958050d63` | 频带关系内容版<br>chbmit-paclock_pt_large | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.652204 |  | 0.935548 | 0.516275 |  |
| `8f91f74dee6d933a6c8f` | 频带关系内容版<br>chbmit-crofremo_n1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.677321 |  | 0.933183 | 0.628506 |  |
| `36ed42c950e5aa42ed13` | 频带关系内容版<br>chbmit-crofremo_s1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.617681 |  | 0.937336 | 0.603144 |  |
| `0ae53a8ab1eeeda15c48` | 频带关系内容版<br>chbmit-paclock_full | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.426605 |  | 0.860164 | 0.568694 |  |
| `323635a76c3ccbd3c556` | 频带关系内容版<br>chbmit-paclock_full | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.449390 |  | 0.879662 | 0.500000 |  |
| `7b37102e6034123c65a9` | 频带关系内容版<br>chbmit-paclock_full | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.341746 |  | 0.807905 | 0.509058 |  |
| `8c46b1bdbad07c1c06f2` | 频带关系内容版<br>chbmit-paclock_full | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.413300 |  | 0.841983 | 0.534303 |  |
| `b9edf307716cee7cc04f` | 频带关系内容版<br>chbmit-paclock_full | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.326600 |  | 0.831068 | 0.500000 |  |
| `41d0566d67dd3a5b1a37` | 频带关系内容版<br>chbmit-paclock_p200_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.427372 |  | 0.859211 | 0.568694 |  |
| `2e35cdfd07cf190d86aa` | 频带关系内容版<br>chbmit-paclock_pac | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.500141 |  | 0.879285 | 0.523463 |  |
| `36b336072da50e435cbe` | 频带关系内容版<br>chbmit-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.510036 |  | 0.909055 | 0.554202 |  |
| `6dc0195a103b0f9bef5b` | 频带关系内容版<br>chbmit-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.417894 |  | 0.904125 | 0.514493 |  |
| `cc38e2503a2ed930e5d4` | 频带关系内容版<br>chbmit-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.589996 |  | 0.921301 | 0.574158 |  |
| `3eaaf20ed13d82fe2424` | 频带关系内容版<br>chbmit-paclock_v2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.580890 |  | 0.903604 | 0.509058 |  |
| `370c9e2338a63269dbb9` | 频带关系内容版<br>chbmit-paclock_v2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.538206 |  | 0.904467 | 0.510870 |  |
| `696ba90746537a261458` | 频带关系内容版<br>chbmit-paclock_v2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.519981 |  | 0.902537 | 0.547043 |  |
| `ff77a47a1f9bba942488` | 频带关系内容版 / carrier<br>chbmit-nb16_carrier_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.691449 |  | 0.921524 | 0.661086 |  |
| `79e462b81f0ee42fcf56` | 频带波形原始版<br>chbmit-paclock_rawpt_large | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.627944 |  | 0.937895 | 0.550666 |  |
| `a880450cf878b0ba7289` | 频带波形原始版<br>chbmit-cf2_v1d192raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.579684 |  | 0.907881 | 0.566971 |  |
| `ba0645b14b320169e98f` | 频带波形原始版<br>chbmit-cf2_v1d192raw | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.687014 |  | 0.920599 | 0.574217 |  |
| `99457bbc646c527df576` | 频带波形原始版<br>chbmit-cf2_v1d256raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.582598 |  | 0.903459 | 0.519928 |  |
| `a9e6b50d436624d04506` | 频带波形原始版<br>chbmit-cf2_v1d256raw | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.655783 |  | 0.907284 | 0.583304 |  |
| `4a21368d05277581fe45` | 频带波形原始版<br>chbmit-cf2_v1raw | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.660577 |  | 0.908816 | 0.563406 |  |
| `370f95f3aa71e38846fc` | 频带波形原始版<br>chbmit-paclock_raw_flat | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.129965 |  | 0.781067 | 0.528196 |  |
| `b1441d7dd6e45d59d4ca` | 频带波形原始版<br>chbmit-paclock_raw_flat | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.156924 |  | 0.809148 | 0.507217 |  |
| `4902fa812617801ad001` | 频带波形原始版<br>chbmit-paclock_raw_flat | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.121088 |  | 0.836522 | 0.503477 |  |
| `18c971189986a8bebbd4` | 频带波形原始版<br>chbmit-paclock_raw_nb1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.326666 |  | 0.873252 | 0.505435 |  |
| `74e490ad21c4e2a6891b` | 频带波形原始版<br>chbmit-paclock_raw_nb1 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.160838 |  | 0.694526 | 0.501812 |  |
| `3fa9a2454d4a7634df2a` | 频带波形原始版<br>chbmit-paclock_raw_nb1 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.247456 |  | 0.760365 | 0.512681 |  |
| `16db211f2806054ad556` | 频带波形原始版<br>chbmit-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.717845 |  | 0.910486 | 0.576029 |  |
| `afedd5895fac195131a3` | 频带波形原始版<br>chbmit-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.604879 |  | 0.914824 | 0.512681 |  |
| `de85e0c287633e35562d` | 频带波形原始版<br>chbmit-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.678800 |  | 0.911359 | 0.552449 |  |

## natural_grid_17224（13 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `57b4041ce2301aa7a90e` | CFC参考替换<br>chbmit-cfft_cfc_reference_20261030 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.351901 | 0.357992 | 0.861868 | 0.609415 |  |
| `ab64ce461f4c65cd78e7` | Candidate C 单路径<br>chbmit-cfft_direct_20261104 | not_explicitly_recorded | 0 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `27a336b8a88fcde6cc7a` | Quad-Range<br>chbmit-cfft_original_control_20261104 | not_explicitly_recorded | 0 | operator_stopped<br>training_progress |  |  |  |  |  |  |
| `45fcaf9777b6c421e7b1-natural` | Quad-Range / identity<br>chbmit-legacy-triaxial-identity-20261105 | not_explicitly_recorded | 2 | interim_eval<br>subset_reevaluation |  | 0.488285 | 0.494156 | 0.880376 | 0.618304 |  |
| `93beac64924f6da0ccbd` | Quad-Range / identity, content:envelope<br>chbmit-eonly-natural-grid-20261108 | frozen_checkpoint | 2 | completed_record<br>clinical_checkpoint_eval |  | 0.534645 | 0.540837 | 0.901655 | 0.596345 |  |
| `cf794bdeef6cca2b19e0` | cbramod<br>cbramod-pretrained | pretrained | 0 | completed_record<br>verified_completion_receipt |  | 0.169454 | 0.170914 | 0.763948 | 0.655150 |  |
| `26fe69635b359bf35504` | cbramod<br>cbramod-scratch | scratch | 0 | completed_record<br>verified_completion_receipt |  | 0.569932 |  | 0.927180 | 0.801287 |  |
| `45b118f024cb26ecdf21` | eegpt<br>chbmit-eegpt_pretrained_native20261025 | pretrained | 0 | completed_record<br>clinical_run |  | 0.532210 | 0.532530 | 0.925391 | 0.666345 |  |
| `59f3b2d67cffa2187533` | eegpt<br>chbmit-eegpt_scratch_official_20261022 | scratch | 0 | completed_record<br>clinical_run |  | 0.379121 | 0.382706 | 0.905464 | 0.622602 |  |
| `920798280e05f1768b5f` | labram<br>chbmit-seed0 | scratch | 0 | completed_record<br>clinical_run |  | 0.267098 |  | 0.842927 | 0.574298 |  |
| `3d4d1e124b13d1ba29eb` | 旧CFC集成（须绑定原源码）<br>chbmit-cfft_cfc_h2_twoaxis_20261029 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.334461 | 0.339741 | 0.839067 | 0.517456 |  |
| `1af758065d5560d29361` | 旧CFC集成（须绑定原源码）<br>chbmit-cfft_cfc_residual_twoaxis_20261030 | not_explicitly_recorded | 0 | incomplete_status_unknown<br>training_progress |  |  |  |  |  |  |
| `9313390815cc919aca52-natural` | 模型未记录<br>chbmit-c-eval-legacy-20261104 | not_explicitly_recorded |  | interim_eval<br>subset_reevaluation |  | 0.317193 | 0.322383 | 0.824056 | 0.613918 |  |

## protocol_not_bound（10 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `dd04193615a6bbb3613d` | biot<br>chbmit-biot_scratch_native20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  |  |  |  |  |  |
| `3973af3406a4ef9cb034` | biot<br>chbmit-biot_pretrained_native20261025 | pretrained | 0 | completed_record<br>clinical_run |  |  |  |  |  |  |
| `b6962b3aa6f727ccf815` | eegconformer<br>chbmit-eegconformer_native_s0 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.255803 | 0.258787 | 0.841830 | 0.652486 |  |
| `dfe949a6909e29c58b5d` | feat_best<br>chbmit-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.046823 |  | 0.711732 | 0.499795 |  |
| `fa7d5ca6204fe4094591` | feat_best<br>chbmit-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.046823 |  | 0.711732 | 0.499795 |  |
| `6fe0cdffa3b3f881dd9d` | feat_best<br>chbmit-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.046823 |  | 0.711732 | 0.499795 |  |
| `7d40a855c48432fe35df` | ffcl<br>chbmit-ffcl_scratch_native20261022 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  |  |  |  |  |  |
| `282e973d0adcc657d95d` | labram<br>chbmit-labram-pretrained-s0 | pretrained | 0 | completed_record<br>clinical_run |  | 0.278963 |  | 0.891284 | 0.587164 |  |
| `bda666a60d8e47da6c92` | reve<br>chbmit-reve_pretrained_rebuild20261021 | pretrained | 0 | completed_record<br>clinical_run |  | 0.466826 | 0.469410 | 0.913467 | 0.718509 |  |
| `46014b4daebebc891ae2` | reve<br>chbmit-reve_scratch_rebuild20261021 | scratch | 0 | completed_record<br>clinical_run |  | 0.325591 | 0.329263 | 0.908594 | 0.617457 |  |
