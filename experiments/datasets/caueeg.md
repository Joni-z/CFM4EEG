# CAUEEG 实验记录

[返回总入口](../README.md)

共 100 条记录。[完整精度及来源 CSV](caueeg.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（100 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `ea143af31437ad23db50` | Duplex<br>caueeg-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.303548 |  |  |  | 0.537753 | 0.527505 |
| `aa3119ed5b06bdfc46aa` | Duplex<br>caueeg-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.311707 |  |  |  | 0.540695 | 0.533988 |
| `a5cab95d319a75410d40` | Duplex<br>caueeg-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.211360 |  |  |  | 0.475347 | 0.464687 |
| `6251cd26d3c45a5fe3b3` | Duplex-A128<br>caueeg-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.185113 |  |  |  | 0.459829 | 0.423564 |
| `28bff0db97de9d8ca0e2` | Duplex-A128<br>caueeg-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.251589 |  |  |  | 0.503327 | 0.497971 |
| `bc1591f00c38e238aa72` | Duplex-A128<br>caueeg-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.303527 |  |  |  | 0.534603 | 0.536335 |
| `16769479eea3927fc48e` | Duplex-A128<br>caueeg-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.298713 |  |  |  | 0.533441 | 0.540593 |
| `da4a8416cb04206bb7e7` | Duplex-A128<br>caueeg-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.260547 |  |  |  | 0.508076 | 0.511352 |
| `f0976bccbf91683c1417` | Duplex-A128<br>caueeg-paclock_tune_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.313932 |  |  |  | 0.542743 | 0.542252 |
| `e3e1954f455d33316c32` | Duplex-A128<br>caueeg-paclock_tune_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.244098 |  |  |  | 0.496255 | 0.498014 |
| `1b3396f86e7a2944f4f6` | Duplex-A128<br>caueeg-paclock_tune_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.336833 |  |  |  | 0.557790 | 0.559621 |
| `515f64d4cdb73d6f260d` | Duplex-F192<br>caueeg-cf2_v1d192_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.229638 |  |  |  | 0.488226 | 0.486061 |
| `c2fa56c97d2fb8d8dd69` | Duplex-F192<br>caueeg-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.263767 |  |  |  | 0.510786 | 0.504827 |
| `9a49bc21e5224b67de68` | Duplex-F192<br>caueeg-cf2_v1d192 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.322099 |  |  |  | 0.549067 | 0.547839 |
| `13d8b9f0a6f1fa4af48c` | Duplex-F192<br>caueeg-cf2_v1d192 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.235852 |  |  |  | 0.492981 | 0.488986 |
| `eed40d824a6ef9097300` | Duplex-F192<br>caueeg-cf2_v1d192_r1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.276118 |  |  |  | 0.519483 | 0.505727 |
| `acb1d67aeb2f94840cf5` | Duplex-F192<br>caueeg-cf2_v1d192_r2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.268866 |  |  |  | 0.513447 | 0.520190 |
| `0284bef3206e5f981f9e` | Duplex-F192<br>caueeg-cf2_v1d192_r3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.196706 |  |  |  | 0.467341 | 0.440263 |
| `d71dc21ec748327c18a4` | Duplex-F192<br>caueeg-cf2_v1d192_r4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.361203 |  |  |  | 0.575371 | 0.564061 |
| `b1674448bc9c77a24c26` | Duplex-F192<br>caueeg-cf2_v3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.313207 |  |  |  | 0.540940 | 0.521953 |
| `fe1091db3995ac691c22` | Quad-Range<br>caueeg-quad_range_20260916 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.316776 |  |  |  | 0.543146 | 0.540559 |
| `ad9386590f6b54d4c00c` | Quad-Range<br>caueeg-range_twoaxis_measured_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.220684 |  |  |  | 0.480661 | 0.468548 |
| `c1513e059f5d5bb2ba92` | Quad-Range<br>caueeg-range_twoaxis_measured_m200_20260917 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.137906 |  |  |  | 0.426273 | 0.390925 |
| `f2fc7d82637e86bf28bd` | Quad16<br>caueeg-quad16_breadth_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.272956 |  |  |  | 0.513416 | 0.510357 |
| `2af0d42742779cf904f7` | Quad16<br>caueeg-quad16_breadth_20260915 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.253418 |  |  |  | 0.503441 | 0.503492 |
| `c93072eae6a1efd0a839` | Quad16<br>caueeg-quad16_breadth_20260915 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.326204 |  |  |  | 0.550817 | 0.534813 |
| `aa7acfc55742aa5636a1` | biot<br>caueeg-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run | 0.324071 |  |  |  | 0.550588 | 0.551944 |
| `ba8d54108886a9e97c56` | biot<br>caueeg-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run | 0.268520 |  |  |  | 0.514147 | 0.517249 |
| `1e48953764f04d75214b` | biot<br>caueeg-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run | 0.273416 |  |  |  | 0.515436 | 0.516243 |
| `d6cb78311a5536ec1d6c` | biot<br>caueeg-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.349756 |  |  |  | 0.566302 | 0.568801 |
| `82b7fa033fbb8d8931ec` | biot<br>caueeg-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.351844 |  |  |  | 0.568828 | 0.552757 |
| `c50f2fc45305bcdceb51` | biot<br>caueeg-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.322396 |  |  |  | 0.547700 | 0.548397 |
| `f016ef8a321d86eb6621` | cbramod<br>caueeg-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.273889 |  |  |  | 0.516140 | 0.521088 |
| `adff2af96f7394c7ca51` | cbramod<br>caueeg-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.317574 |  |  |  | 0.544438 | 0.550535 |
| `5825c73b74b8a829aae1` | cbramod<br>caueeg-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.223475 |  |  |  | 0.482868 | 0.489574 |
| `2ab142e706134a2b5705` | cbramod<br>caueeg-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run | 0.184022 |  |  |  | 0.456883 | 0.457892 |
| `c62a625186d11b19bfb8` | cbramod<br>caueeg-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run | 0.230996 |  |  |  | 0.489506 | 0.469650 |
| `9c09dbaf0f9144fa5513` | cbramod<br>caueeg-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `1e1d2789f927d185cfe2` | cbramod_paclockfe宿主转接<br>caueeg-cbramod_crofremo | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.179638 |  |  |  | 0.455235 | 0.437983 |
| `5ed5a6626575402056a4` | cbramod_paclockfe宿主转接<br>caueeg-cbramod_crofremo | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `911cdca98071b327fa1f` | cbramod_paclockfe宿主转接<br>caueeg-cbramod_crofremo | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.238255 |  |  |  | 0.492301 | 0.494034 |
| `41b73bec635d1d79d789` | cnn_transformer<br>caueeg-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.236928 |  |  |  | 0.492112 | 0.492790 |
| `fde35a5fd246e5104b70` | cnn_transformer<br>caueeg-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.257808 |  |  |  | 0.506359 | 0.500807 |
| `7f7f84a17dedbc48c74a` | cnn_transformer<br>caueeg-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.306637 |  |  |  | 0.537116 | 0.537177 |
| `3fe27d3abc02df301aed` | contrawr<br>caueeg-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.317244 |  |  |  | 0.544094 | 0.534775 |
| `c9ad008bd1931e5067a6` | contrawr<br>caueeg-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.347496 |  |  |  | 0.563979 | 0.522802 |
| `4c26b5d4fa9bfc2a2e30` | contrawr<br>caueeg-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.232168 |  |  |  | 0.488621 | 0.492158 |
| `77bcbb3eeea6a3485ff3` | csbrain<br>caueeg-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.234520 |  |  |  | 0.488942 | 0.491918 |
| `ee1d10926df2c0d3ae4d` | csbrain<br>caueeg-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.291408 |  |  |  | 0.527905 | 0.531154 |
| `110a192970cb602f1954` | csbrain<br>caueeg-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.204108 |  |  |  | 0.470280 | 0.475182 |
| `afd24f57c5dd4abaa430` | eegconformer<br>caueeg-eegconformer_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.210452 |  |  |  | 0.475356 | 0.469384 |
| `6a9612976e97abfc4c1c` | eegconformer<br>caueeg-eegconformer_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.268277 |  |  |  | 0.514001 | 0.507839 |
| `e5bb6f7ac3c82f767843` | eegconformer<br>caueeg-eegconformer_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.174587 |  |  |  | 0.451124 | 0.437826 |
| `e4e7441ae3589502612a` | eegconformer<br>caueeg-eegconformer_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.216914 |  |  |  | 0.477259 | 0.485204 |
| `d8022cb32577dac21575` | eegnet<br>caueeg-eegnet_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.275981 |  |  |  | 0.516221 | 0.522034 |
| `db2259f771f366e4a316` | eegnet<br>caueeg-eegnet_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.278149 |  |  |  | 0.517463 | 0.521568 |
| `45848c325c4c5381e5ae` | eegnet<br>caueeg-eegnet_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.249959 |  |  |  | 0.501759 | 0.501043 |
| `99d747d66d1be64f415e` | eegnet<br>caueeg-eegnet_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.317702 |  |  |  | 0.548047 | 0.543671 |
| `a75a446f4822980bcb90` | eegpt<br>caueeg-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.322451 |  |  |  | 0.550015 | 0.552042 |
| `a10d0d57a78facc40628` | eegpt<br>caueeg-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.283571 |  |  |  | 0.524487 | 0.526834 |
| `8dc146ad2d49ea4fcdaf` | eegpt<br>caueeg-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.350001 |  |  |  | 0.566731 | 0.575126 |
| `690941120e843df262e8` | eegpt<br>caueeg-eegpt_scratch | scratch | 0 | completed_record<br>clinical_run | 0.024549 |  |  |  | 0.350265 | 0.226565 |
| `851e83f2b4c873ad8885` | eegpt<br>caueeg-eegpt_scratch | scratch | 1 | completed_record<br>clinical_run | 0.145624 |  |  |  | 0.435160 | 0.345158 |
| `130df4dfe8a265af084b` | eegpt<br>caueeg-eegpt_scratch | scratch | 2 | completed_record<br>clinical_run | 0.239139 |  |  |  | 0.495343 | 0.478035 |
| `6150bf6891a76605702d` | feat_best<br>caueeg-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.245794 |  |  |  | 0.498653 | 0.489191 |
| `73eeef9b70adfa0a8c75` | feat_best<br>caueeg-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.245794 |  |  |  | 0.498653 | 0.489191 |
| `ea3b9842f52dcd2e5e08` | feat_best<br>caueeg-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.245794 |  |  |  | 0.498653 | 0.489191 |
| `f9481ab2c517ff7da5e1` | ffcl<br>caueeg-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.335858 |  |  |  | 0.559912 | 0.556783 |
| `07cb56bedd7f5d1819a8` | ffcl<br>caueeg-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.325898 |  |  |  | 0.550570 | 0.554446 |
| `0b94257929f4756897f4` | ffcl<br>caueeg-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.338862 |  |  |  | 0.560180 | 0.533724 |
| `969fe19cc0f81624b512` | labram<br>caueeg-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `9ee7da0fa74a1e01ac01` | labram<br>caueeg-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `9e5da3e53beef28cc242` | labram<br>caueeg-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `0acd89c837dbd125d417` | labram<br>caueeg-labram_pt_lr1e4 | pretrained | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `b096f39dd74b8bfc68ed` | labram<br>caueeg-labram_pt_lr2e4 | pretrained | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `3ba86c541f758db11bc8` | labram<br>caueeg-labram_pt_lr5e5 | pretrained | 0 | completed_record<br>clinical_run | 0.000000 |  |  |  | 0.333333 | 0.168848 |
| `e2e4593efa3bce9a86dd` | labram<br>caueeg-labram_scratch | scratch | 0 | completed_record<br>clinical_run | 0.248305 |  |  |  | 0.500887 | 0.498149 |
| `3a7a5bdc885c9315abeb` | labram<br>caueeg-labram_scratch | scratch | 1 | completed_record<br>clinical_run | 0.249945 |  |  |  | 0.500907 | 0.503536 |
| `120ec3cd35db7c496caf` | labram<br>caueeg-labram_scratch | scratch | 2 | completed_record<br>clinical_run | 0.276669 |  |  |  | 0.518577 | 0.524097 |
| `9dc7998d76fbbe4c1721` | reve<br>caueeg-reve_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.145207 |  |  |  | 0.430311 | 0.426499 |
| `fccbd36d3ed04457125c` | reve<br>caueeg-reve_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.183280 |  |  |  | 0.455729 | 0.459769 |
| `75dda7cd8bfed6f6564a` | reve<br>caueeg-reve_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.103837 |  |  |  | 0.403495 | 0.408217 |
| `04630108cbf91f54647b` | sparcnet<br>caueeg-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.143865 |  |  |  | 0.429827 | 0.430237 |
| `188aba082fcba5dc7033` | sparcnet<br>caueeg-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.101663 |  |  |  | 0.401893 | 0.405948 |
| `b6a3a318d66210f16a86` | sparcnet<br>caueeg-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.238476 |  |  |  | 0.492379 | 0.497056 |
| `b2ec9e00a53b92f176dd` | st_transformer<br>caueeg-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.269661 |  |  |  | 0.512625 | 0.511455 |
| `051672cd7a3e911612f8` | st_transformer<br>caueeg-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.206975 |  |  |  | 0.470788 | 0.476104 |
| `941aa1dfa24d5011665d` | st_transformer<br>caueeg-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.293667 |  |  |  | 0.530516 | 0.529275 |
| `ae67907d1ce83b0ad898` | tfm<br>caueeg-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run | 0.339310 |  |  |  | 0.559102 | 0.563799 |
| `fc26e60b74b1b9717925` | tfm<br>caueeg-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run | 0.293579 |  |  |  | 0.529991 | 0.533620 |
| `8191801ffd602295e36f` | tfm<br>caueeg-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run | 0.238852 |  |  |  | 0.493642 | 0.499253 |
| `578f4ea94399b471a7a1` | tfm<br>caueeg-tfm_scratch | scratch | 0 | completed_record<br>clinical_run | 0.256165 |  |  |  | 0.504184 | 0.506338 |
| `dd7d6cae2609528e97f4` | tfm<br>caueeg-tfm_scratch | scratch | 1 | completed_record<br>clinical_run | 0.216686 |  |  |  | 0.478977 | 0.474556 |
| `b124ca5245f69aef268e` | tfm<br>caueeg-tfm_scratch | scratch | 2 | completed_record<br>clinical_run | 0.284082 |  |  |  | 0.522947 | 0.526245 |
| `d418f2b753b251cbba91` | 频带关系内容版<br>caueeg-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.178097 |  |  |  | 0.454279 | 0.446961 |
| `92b7bab5f21ab9c35c3b` | 频带关系内容版<br>caueeg-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.277495 |  |  |  | 0.520672 | 0.510856 |
| `467e7625c96e09f9038c` | 频带关系内容版<br>caueeg-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.244836 |  |  |  | 0.496362 | 0.497555 |
| `ce247d7a93f0d806eac4` | 频带波形原始版<br>caueeg-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run | 0.279076 |  |  |  | 0.518892 | 0.526023 |
| `5d4a0ef70dfec51cfe68` | 频带波形原始版<br>caueeg-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run | 0.226004 |  |  |  | 0.484450 | 0.488839 |
| `612b0d74b7b00ecc2161` | 频带波形原始版<br>caueeg-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run | 0.325263 |  |  |  | 0.548525 | 0.548544 |
