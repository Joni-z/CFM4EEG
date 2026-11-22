# CroFreMo：高阶统计与双谱方向阅读及定位
日期：2026-10-09

## 结论
保留现有 relation-conditioned tokenizer 和两轴 encoder。以 higher-order statistics 作为动机，以 polyspectral analysis / bispectrum 解释功率谱不能充分确定的跨频结构；暂不把当前测量算子命名为 bispectral estimator。
现阶段的新增工作重点是测量含义诊断和“同一关系作为特征 / 作为内容变换”的直接比较，不能用换术语替代这些证据。

## 阅读范围
1、2、4、5、6、8 的全文已取得，重点核对方法、实验及讨论；NeuroRVQ 同时查看了 2025 v1 与目前新版。第 3 篇核对出版商公开的方法、结果和讨论内容，未下载完整 PDF；第 7 篇仅能读到出版商的摘要、引言和方法片段，完整实验和补充材料尚不可核查。
另读 Bispectral Neural Networks (ICLR 2023)、Aru et al. 的 CFC 方法学讨论，并查阅 PyBispectra 论文/官方接口。
本笔记不更新论文数据，不启动训练。

## 八篇文章的作用

### 1. Kovach et al., 2018
https://arxiv.org/abs/1705.10435
https://doi.org/10.1016/j.neuroimage.2018.02.033

核心是常见 PAC 测量与平滑双谱之间的联系；Appendix C 明确推导 phase–power coherence。滤波器核、带宽与估计形式影响对应关系。双谱同时包含嵌套振荡、非正弦波形和瞬态的信息，不能直接解释成两个独立振荡器通信。
用途：连接 PAC 启发的设计与高阶谱背景。不能据此宣称当前单位相位乘中心化幅度的算子严格等于三阶谱。

### 2. The influence of nonlinear resonance..., 2026
https://www.nature.com/articles/s42003-026-10164-5

BiSCA 联合分析功率谱和双相干，区分非周期背景与振荡成分。数据为 960 名健康头皮 EEG 受试者及 106 名患者的 1,771 个颅内通道。报告的非线性或非高斯统计结构比例不是所有通道上存在生理 PAC 的比例。方法主要分析静息态离散谐波关系；扩展到宽带 theta–gamma PAC 被列为未来工作。
用途：为 EEG 确有功率谱之外的结构提供实证动机。不能推出该结构必然有助分类，也不能借用其生理解释来认证当前短窗口测量。

### 3. Low-rank tensor decomposition..., 2026
https://doi.org/10.1016/j.jneumeth.2026.110739
https://www.sciencedirect.com/science/article/abs/pii/S0165027026000695

方法针对传感器三元组的跨双谱张量，用共享实值空间混合矩阵和复值源交互核心压缩，配合源重建与 MOCA 解混。包含仿真和静息 EEG 示例。它解决源级解释与高维统计压缩，不是 tokenizer 分类。
用途：紧凑建模高阶结构的先例。其源混合低秩假设不能直接转移为我们的频带关系矩阵低秩保证，也没有理由为 16 带前端立即加入传感器三阶张量。

### 4. Multidimensional bicoherence / MACB, 2024
https://www.nature.com/articles/s41598-024-57014-0

MACB 为向量源信号提供多维反对称跨双相干，关注二次相位耦合和瞬时混合干扰。旋转不变性是源空间坐标的正交变换不变性。主要验证是三组仿真，实际 EEG/MEG 应用是后续方向。
用途：提醒区分统计依赖、混合和源交互，借鉴受控仿真。其坐标不变性与本项目的解析相位参考变换不同，不能互换定理。

### 5. TFM-Tokenizer, ICLR 2026
https://arxiv.org/abs/2502.16060
https://www.jathurshanpradeepkumar.com/TFM-Tokenizer/

双路时频编码、频率 Transformer、门控聚合和 VQ 学习 motif。它在 tokenizer 内已经学习频带间依赖，因此“已有 tokenizer 都不处理跨频关系”不成立。冻结预训练 tokenizer 后再做 encoder 的 masked-token pretraining；BIOT/LaBraM 替换也涉及宿主训练流程。
用途：最近的任务和 tokenizer 对手。差别应写成“通用学习得到的时频依赖”与“显式测量的特定跨频统计量参与内容变换”，不能写成有/无跨频关系。其预训练实验与我们从头监督训练的换头不能直接等同。

### 6. NeuroRVQ, 2025 v1 / 后续新版
https://arxiv.org/abs/2510.13068v1
https://arxiv.org/abs/2510.13068

多尺度卷积、残差向量量化，以及相位和幅度感知的重建目标。新版扩展到多类 biosignals，加入更明确的相位方向损失和相关分析。
用途：证明 tokenizer 的结构与损失可以影响频率信息保留。重建保真度与特定跨频关系显式构造是不同目标，后者可能互补。其相位重建不能被描述成“不包含相位”。

### 7. DyST / BRAVE, 2026（访问范围受限）
https://doi.org/10.1016/j.bspc.2026.110765
https://www.sciencedirect.com/science/article/abs/pii/S1746809426013194

可读部分描述：GFP 汇集多通道活动，经 VMD 分解和相位转折做动态分段；按频带路由到专门 codebook。主要问题是 token 边界和不同频带的表示。
用途：tokenizer 中已有生理/信号结构先验的先例。我们回答的是关系如何改变内容表示。未核查全部实验，不根据其摘要的最大增益为本项目设数值目标。

### 8. Bispectrum + LSTM, 2019
https://www.nature.com/articles/s41598-019-52152-2

三只犬的 16 通道颅内 EEG，30 秒窗口计算双谱，提取两类熵特征，用 5 分钟序列训练受试体内 LSTM，按时间和发作划分训练/验证/测试。任务为发作前与发作间分类。
用途：高阶谱特征可进入神经分类器的直接先例，排除“首次将高阶统计用于 EEG 深度学习”的定位。其结果不能直接外推到我们的头皮 EEG 发作检测。

## 额外必须认识的先例
- Bispectral Neural Networks, ICLR 2023：https://arxiv.org/abs/2209.03416
  已经把双谱形式做成可学习网络原语，研究群不变表示。我们的贡献不能只说“把高阶统计写进网络”。它的完整不变量性质不能自动套到局部、归一化、投影后的 EEG tokenizer。
- Aru et al., 2015：https://arxiv.org/abs/1405.7965
  非平稳、共同输入、波形和滤波能影响 CFC；相位随机化也会破坏非平稳和波形结构，不能被当成只去 PAC 的干预。
- PyBispectra：https://joss.theoj.org/papers/10.21105/joss.08504
  可作独立统计参照，不是训练模块的现成替代品。官方文档：https://pybispectra.readthedocs.io/latest/

## 对照代码后的判断
核对了 MBP code/twoaxis-runtime/paclock_bench/models/paclock/frontend/triaxial.py 及滤波器实现。

1. Z 使用单位解析相位和中心化幅度，再进行局部平均。它是非线性变换信号上的统计量，不是未经推导便可命名的原波形三阶累积量。
2. 关系幅度决定相对混合权重，关系方向参与对齐，随后旋转 envelope / waveform 特征对。当前设计最明确的独特操作在这一步。
3. token patch 与 moment window 已解耦；无需为新叙事增加新的 encoder。现有长窗口结果属于原有编码器/配方时，不可自动当作当前两轴模型的参数选择依据。
4. 主文稿已含有限 FIR 响应、低频重叠和权重富集分析。应复用相关脚本/记录，按新问题汇总，避免重复做同一张图。
5. 旧 concat 分支只拼接 amplitude_feat 和 aligned_phase 的实虚部；构造器还要求 waveform modulation 与 rotation 同用。它不能仅改一个开关就作为当前 quadrature 模型的公平 PAC-as-feature 对照。
6. identity 保留分配的相位参数，但相位支路失活。因此“参数张量数量相同”不等于“有效可训练路径完全相同”。涉及算子优于普通特征的结论还需要内容和容量更匹配的对照。
7. 代码注释记载过多尺度窗口、显著性门控和 concat 尝试；重启前应核查相应真实运行记录，不能只凭注释认定成功或失败。

## 推荐定位与叙事
定位：以高阶统计为动机的 relation-conditioned EEG tokenization。

叙事顺序：
1. EEG tokenization 的问题：将多通道信号变成便于后续学习的有限表示。
2. 现有方法研究波形/谱重建、时频 motif、多尺度或自适应分段；其中部分已学习频带关系。
3. 统计依据：功率谱不能完全确定非高斯信号的跨频结构；双谱为一类相位关联提供具体例子。
4. 研究问题：能否把选定的跨频依赖提前测出，用它直接参与 token 的构造？
5. 实现：PAC 启发的局部测量控制 envelope 与 waveform 特征对的旋转，后续 encoder 学习任务相关的时空组合。
6. 证据：同信息的不同使用方式比较；跨任务表现和宿主适配说明范围。
7. 边界：不是普遍优于任意 encoder，也不将统计依赖解释成已识别的生理通信。

higher-order statistics 是背景；polyspectral analysis 是具体理论联系；relation-conditioned tokenization 是方法贡献。三个层次不要混写成同一个已证明的命题。
“二阶统计不足”本身不能证明波形神经 tokenizer 丢了该信息，或证明关系只能在 tokenizer 中学习。后两者是待实验检验的设计假设。

## 最小新增证据

### A. 统计含义诊断，先不重训大模型
构造已知耦合、非耦合、非正弦单源三类信号，控制功率谱、噪声、滤波重叠和窗口长度。比较当前 moment 与独立双谱/双相干估计，分别检查强度与方向。
用于确认当前算子在哪些条件下感知目标依赖，在哪些条件下对波形/滤波也有响应。
可用匹配 Fourier 幅度、改变相位关系的样本做直观示例；普通相位随机化不被解释成 PAC 专属干预。
真实 EEG 上用相同滤波和窗口做参照图，不要求两个不同估计器数值相等；解释一致性和失败区域。

### B. 同一关系量的使用方式，核心训练对照
固定两轴 encoder、16 带、内容支路、token 数、数据划分、增强、训练预算及验证选点：
- identity：保留同一内容，不做关系调制。
- relation-as-feature：同一组关系信息与相同 envelope / waveform 内容一并输入投影。
- relation-conditioned：现有旋转构造。
在 TUEV、CHB-MIT 先做一组固定 seed 的开发试验；只有配置和配方完全可比的已有运行才能复用。确认实现后用同一组 seed 汇总配对差，不能各臂选最好测试分数。
特征组应明确暴露的是原始矩、归一化系数还是最终对齐方向；若使用最终方向，只能检验“有/无乘性作用”，不能声称已对比所有 PAC-as-feature 方法。
新投影的有效参数数目、FLOPs 应报告并尽量匹配；不要通过移除其 waveform 支路来凑参数。
若两者相当，论点停在显式跨频统计有用；若旋转更好，才能把使用方式写成经验证优势。

### C. 估计时间尺度，仅补缺少的当前模型对照
短 token 保持不动，用已有 0.25/1/5 秒经验筛选一个 longer-window 候选，优先 1 秒作为候选而不是已定最优。不要将长窗口结果直接变为更精确生理 PAC 的证据。
多尺度只在单尺度证据显示跨任务冲突且收益值得额外复杂度时再考虑。

### D. 下一个阶段，条件触发
若要直接声称 bispectral tokenizer，再替换测量支路为有明确定义的三频乘积或可推导的三阶矩，控制频率和关系、估计平均、幅度归一化、噪声方差。其余内容支路/encoder 可保持。
可先在合成数据和少数真实样本评估估计质量，再考虑监督训练；不应先铺全部数据集。
不为叙事同时引入 RVQ、源定位低秩张量、大规模预训练或新 encoder。

## 判定
- 统计诊断不支持当前 Z 承载拟议结构：先修测量/窗口，或缩小定位。
- 关系构造相对 identity 没有稳定收益：不能通过高阶统计文献替代负的任务证据。
- 显式特征和旋转都改善、两者相当：强调显式统计关系，不强调旋转形式占优。
- 旋转在配对控制中额外改善：支持 relation-conditioned construction。
- 宿主间有正负差异：报告适用范围，不能概括为即插即用的普遍增益。

