# siena 实验记录

[返回总入口](../README.md)

共 95 条记录。[完整精度及来源 CSV](siena.csv)。
同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。
空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。

## protocol_not_bound（95 条）

| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `d649382f928f64048e23` | Duplex<br>siena-cf2_v0d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.239320 |  | 0.836048 | 0.500000 |  |
| `5daebda59938358438f4` | Duplex<br>siena-cf2_v0d192cs | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.398188 |  | 0.851536 | 0.679044 |  |
| `4ebfa04426a4ebcedcf3` | Duplex<br>siena-cf2_v1d256 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.269143 |  | 0.864586 | 0.500000 |  |
| `3ddf6f6e82a419615dac` | Duplex-A128<br>siena-paclock_duplex_ptF | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.468211 |  | 0.835566 | 0.549180 |  |
| `71d44677ce1440587577` | Duplex-A128<br>siena-paclock_duplex_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.421666 |  | 0.862174 | 0.500000 |  |
| `c3b1a8499c833d6fbb37` | Duplex-A128<br>siena-paclock_duplex_ptS | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.491169 |  | 0.829270 | 0.647495 |  |
| `aedb87e558c1b39caa20` | Duplex-A128<br>siena-paclock_duplex_ptS | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.411203 |  | 0.848822 | 0.647174 |  |
| `dc77fb3235b5ddce4297` | Duplex-A128<br>siena-paclock_duplex_ptS | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.466935 |  | 0.829016 | 0.500000 |  |
| `08d9c440cb538a335f45` | Duplex-A128<br>siena-paclock_ptFtune_t1 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.467765 |  | 0.835082 | 0.549180 |  |
| `ab2881aa8651495b05bc` | Duplex-A128<br>siena-paclock_ptFtune_t2 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.329010 |  | 0.841040 | 0.500000 |  |
| `c1ff922f55809c84b440` | Duplex-A128<br>siena-a128_r1_20260916 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.096272 |  | 0.858685 | 0.500000 |  |
| `47e6a3901c7612c3272d` | Duplex-A128<br>siena-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.132906 |  | 0.868334 | 0.500000 |  |
| `c74b4586ed37c023dcff` | Duplex-A128<br>siena-paclock_duplex | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.109804 |  | 0.870210 | 0.500000 |  |
| `157adc965ac7e7bf23f9` | Duplex-A128<br>siena-paclock_duplex | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.268896 |  | 0.869036 | 0.500000 |  |
| `0bde8d4d7d9c05b3ee62` | Duplex-A128<br>siena-paclock_duplex | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.131800 |  | 0.867226 | 0.500000 |  |
| `af4aec5462ee5d427aec` | Duplex-A128<br>siena-paclock_duplex_b32x4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.065443 |  | 0.833905 | 0.500000 |  |
| `daa4926616917a1804ce` | Duplex-F192<br>siena-cf2_v1d192_ptR | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.342217 |  | 0.841565 | 0.500000 |  |
| `a8644204a6b88408edf3` | Duplex-F192<br>siena-cf2_v1d192 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.200920 |  | 0.869882 | 0.500000 |  |
| `38a92e721d6c795d538f` | Duplex-F192<br>siena-cf2_v1d192 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.254387 |  | 0.870302 | 0.500000 |  |
| `b8ae5d7719c6e78c23c5` | Duplex-F192<br>siena-cf2_v1d192 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.204202 |  | 0.873942 | 0.500000 |  |
| `b893615861d68c3a624e` | Duplex-F192<br>siena-cf2_v3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.279956 |  | 0.845193 | 0.677898 |  |
| `c586000587e3b36d4a1d` | Quad16<br>siena-quad16_breadth_20260915 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.345036 |  | 0.865858 | 0.500000 |  |
| `ada6715dadb8c1a8804f` | Quad16<br>siena-quad16_breadth_20260915 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.219894 |  | 0.843381 | 0.500000 |  |
| `efbfe595f4ebf41db731` | Quad16<br>siena-quad16_breadth_20260915 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.295514 |  | 0.851339 | 0.500000 |  |
| `3a650ad7f9afdddc005e` | biot<br>siena-biot_prest16 | checkpoint_loaded | 0 | completed_record<br>clinical_run |  | 0.068769 |  | 0.822787 | 0.500000 |  |
| `95e0ce40ca69c7f03c84` | biot<br>siena-biot_prest16 | checkpoint_loaded | 1 | completed_record<br>clinical_run |  | 0.102621 |  | 0.768411 | 0.508197 |  |
| `0329ef1447f08ee40570` | biot<br>siena-biot_prest16 | checkpoint_loaded | 2 | completed_record<br>clinical_run |  | 0.100011 |  | 0.794865 | 0.532328 |  |
| `8e4f096d97383655cb1a` | biot<br>siena-biot_scratch | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.158414 |  | 0.752700 | 0.540984 |  |
| `d1e3d6a53b9f6aa6aade` | biot<br>siena-biot_scratch | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.214974 |  | 0.802805 | 0.565299 |  |
| `36ed466e34afb34c354d` | biot<br>siena-biot_scratch | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.126471 |  | 0.791680 | 0.540708 |  |
| `2b4ca05ecf4cccf2fe2c` | cbramod<br>siena-cbramod_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.352980 |  | 0.825943 | 0.565528 |  |
| `0c271afb8d4edcdbe8d4` | cbramod<br>siena-cbramod_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.444123 |  | 0.853610 | 0.696309 |  |
| `e2515284bf03deb917db` | cbramod<br>siena-cbramod_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.475755 |  | 0.815111 | 0.712702 |  |
| `4822ccecfed5f7958240` | cbramod<br>siena-cbramod_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.269271 |  | 0.869727 | 0.638106 |  |
| `7687c8722b8538a32423` | cbramod<br>siena-cbramod_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.160691 |  | 0.782697 | 0.532695 |  |
| `18ae1a9431f037fe85c5` | cbramod<br>siena-cbramod_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.167881 |  | 0.779192 | 0.565390 |  |
| `3d8f3f422b5ccb711026` | cbramod_paclockfe宿主转接<br>siena-cbramod_crofremo | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.042568 |  | 0.825352 | 0.500000 |  |
| `e194246d01af691c2988` | cbramod_paclockfe宿主转接<br>siena-cbramod_crofremo | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.181510 |  | 0.843753 | 0.500000 |  |
| `7c7eb599d2a2f21fb528` | cbramod_paclockfe宿主转接<br>siena-cbramod_crofremo | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.042835 |  | 0.814660 | 0.500000 |  |
| `6d0054af4191e897d8a2` | cnn_transformer<br>siena-cnn_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.480788 |  | 0.874421 | 0.721036 |  |
| `6dc32c6b489d66331b76` | cnn_transformer<br>siena-cnn_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.307256 |  | 0.833465 | 0.670389 |  |
| `3adde2a628a4d45ab8d2` | cnn_transformer<br>siena-cnn_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.480087 |  | 0.867131 | 0.713023 |  |
| `ae049f55f6214bcb64dd` | contrawr<br>siena-contrawr | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.379959 |  | 0.798361 | 0.670801 |  |
| `7e8e507e0cd371a6f0c2` | contrawr<br>siena-contrawr | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.487265 |  | 0.836052 | 0.631148 |  |
| `c015c5870dbf4f128ac7` | contrawr<br>siena-contrawr | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.486755 |  | 0.800514 | 0.742921 |  |
| `77d2a0e69d56cd4f6886` | csbrain<br>siena-csbrain_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.140173 |  | 0.719241 | 0.564748 |  |
| `8bec6df464cba71ccbda` | csbrain<br>siena-csbrain_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.353098 |  | 0.842872 | 0.549180 |  |
| `25791e035d3ff5aeb011` | csbrain<br>siena-csbrain_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.205747 |  | 0.626398 | 0.597627 |  |
| `0efa89877ae245379d88` | eegconformer<br>siena-eegconformer_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.537597 |  | 0.837740 | 0.704872 |  |
| `5be4c2c3dc35e8e4d147` | eegconformer<br>siena-eegconformer_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.520802 |  | 0.850293 | 0.688525 |  |
| `99ddb4122abb3d9132f9` | eegconformer<br>siena-eegconformer_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.472012 |  | 0.812531 | 0.672131 |  |
| `71eefa57d87a5e6b668c` | eegconformer<br>siena-eegconformer_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.506778 |  | 0.813083 | 0.704918 |  |
| `ecffbcdefcaa2907e09b` | eegnet<br>siena-eegnet_t1 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.385451 |  | 0.872026 | 0.742187 |  |
| `942d46d85f3b6143bf14` | eegnet<br>siena-eegnet_t2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.234207 |  | 0.875524 | 0.784535 |  |
| `07e9ecafd04feec4356c` | eegnet<br>siena-eegnet_t3 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.343050 |  | 0.828333 | 0.670480 |  |
| `ac3daa90e2a787fa4772` | eegnet<br>siena-eegnet_t4 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.285065 |  | 0.840504 | 0.666720 |  |
| `f3578bc3b8e430c9f6df` | eegpt<br>siena-eegpt_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.238966 |  | 0.836254 | 0.557377 |  |
| `d14194c17f26c3663ac6` | eegpt<br>siena-eegpt_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.364777 |  | 0.846787 | 0.500000 |  |
| `3ba70ab5afcef07d8f59` | eegpt<br>siena-eegpt_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.391383 |  | 0.855691 | 0.500000 |  |
| `f325c05bfddb07f637be` | eegpt<br>siena-eegpt_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.110700 |  | 0.873809 | 0.500000 |  |
| `c917923b8c5016daa272` | eegpt<br>siena-eegpt_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.110341 |  | 0.874908 | 0.500000 |  |
| `ba1d76a9884fc159cedc` | eegpt<br>siena-eegpt_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.163973 |  | 0.844837 | 0.508197 |  |
| `f61be50a346bd2bd1d95` | feat_best<br>siena-feat_best | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.040440 |  | 0.722663 | 0.709551 |  |
| `9335f7e297c4e53d6798` | feat_best<br>siena-feat_best | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.040440 |  | 0.722663 | 0.709551 |  |
| `0c4aee20cff64c3af313` | feat_best<br>siena-feat_best | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.040440 |  | 0.722663 | 0.709551 |  |
| `4ca317804c3993c3519d` | ffcl<br>siena-ffcl | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.308351 |  | 0.820734 | 0.614341 |  |
| `dd0c12914d8c248b6b84` | ffcl<br>siena-ffcl | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.449156 |  | 0.803802 | 0.696630 |  |
| `7badf0ff016f0e6e03dd` | ffcl<br>siena-ffcl | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.506589 |  | 0.830707 | 0.704826 |  |
| `130c37499f95550e7013` | labram<br>siena-labram_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.312427 |  | 0.794964 | 0.654087 |  |
| `65e901cda9ca83da2912` | labram<br>siena-labram_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.280690 |  | 0.763596 | 0.646349 |  |
| `e704c5fe21ef8b175e21` | labram<br>siena-labram_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.314908 |  | 0.744589 | 0.573770 |  |
| `54ac556678b62a27bb43` | labram<br>siena-labram_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.499460 |  | 0.847603 | 0.500000 |  |
| `eae94cb3a25631874292` | labram<br>siena-labram_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.496568 |  | 0.823481 | 0.500000 |  |
| `b799a018388aaf4c1dd4` | labram<br>siena-labram_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.477873 |  | 0.838791 | 0.500000 |  |
| `e8f9ea3d9b94b60e10ac` | reve<br>siena-reve_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.382794 |  | 0.856673 | 0.743655 |  |
| `fd254d6479c1ad4811ac` | reve<br>siena-reve_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.591173 |  | 0.857311 | 0.784226 |  |
| `9aa0da8c6527d663582a` | reve<br>siena-reve_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.580421 |  | 0.855525 | 0.737659 |  |
| `fc6e636a50d1d0e63f06` | sparcnet<br>siena-sparcnet | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.405655 |  | 0.797352 | 0.663522 |  |
| `366e6b4a7b44ef202b23` | sparcnet<br>siena-sparcnet | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.329115 |  | 0.761047 | 0.622905 |  |
| `f20b98cdbf7a01b12620` | sparcnet<br>siena-sparcnet | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.336479 |  | 0.843474 | 0.630597 |  |
| `c707a7185a5386a20c6d` | st_transformer<br>siena-st_transformer | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.004239 |  | 0.381301 | 0.500000 |  |
| `298fcf8a46483c18009a` | st_transformer<br>siena-st_transformer | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.332882 |  | 0.837749 | 0.630414 |  |
| `0a1f1827a323bd87f0c8` | st_transformer<br>siena-st_transformer | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.316210 |  | 0.776202 | 0.662650 |  |
| `9900722b300b828ba8fd` | tfm<br>siena-tfm_pretrained | pretrained | 0 | completed_record<br>clinical_run |  | 0.069274 |  | 0.807150 | 0.499954 |  |
| `8225af41edea9c0d729b` | tfm<br>siena-tfm_pretrained | pretrained | 1 | completed_record<br>clinical_run |  | 0.047107 |  | 0.720134 | 0.500000 |  |
| `25e0335b8609ca6982b4` | tfm<br>siena-tfm_pretrained | pretrained | 2 | completed_record<br>clinical_run |  | 0.064474 |  | 0.784881 | 0.516072 |  |
| `a7374bf7cec5a64c6c85` | tfm<br>siena-tfm_scratch | scratch | 0 | completed_record<br>clinical_run |  | 0.007805 |  | 0.588655 | 0.500000 |  |
| `3ee516dc1601cefbed78` | tfm<br>siena-tfm_scratch | scratch | 1 | completed_record<br>clinical_run |  | 0.022172 |  | 0.518031 | 0.500000 |  |
| `47312d219d2ab2a4986c` | tfm<br>siena-tfm_scratch | scratch | 2 | completed_record<br>clinical_run |  | 0.005914 |  | 0.538038 | 0.500000 |  |
| `20326ce02e5ce085370e` | 频带关系内容版<br>siena-paclock_rot2 | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.377687 |  | 0.838467 | 0.606557 |  |
| `848e9d376167b789087c` | 频带关系内容版<br>siena-paclock_rot2 | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.232406 |  | 0.857111 | 0.500000 |  |
| `e61e340254872498d676` | 频带关系内容版<br>siena-paclock_rot2 | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.332352 |  | 0.847326 | 0.500000 |  |
| `8c4977276c8484e512e4` | 频带波形原始版<br>siena-paclock_rawtok | not_explicitly_recorded | 0 | completed_record<br>clinical_run |  | 0.346262 |  | 0.834771 | 0.687424 |  |
| `6fe7f453d87ee94b0a08` | 频带波形原始版<br>siena-paclock_rawtok | not_explicitly_recorded | 1 | completed_record<br>clinical_run |  | 0.023983 |  | 0.802070 | 0.499220 |  |
| `74c2190d655bd9e8b6b2` | 频带波形原始版<br>siena-paclock_rawtok | not_explicitly_recorded | 2 | completed_record<br>clinical_run |  | 0.142632 |  | 0.837587 | 0.500000 |  |
