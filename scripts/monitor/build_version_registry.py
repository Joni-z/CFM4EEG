"""Targeted version inventory; preserves original run identities and paper tables."""
import datetime,hashlib,json,statistics,subprocess
from pathlib import Path
BASE=Path('/work1/chenyuyou/yifanwang/Zhizhe')
OUT=BASE/'CFM4EEG-anchored-watch-20260915'
VERSIONS=[
 ('Duplex-A128','早期双路版','paclock_duplex','PACLock','8 频段，D128，显式频段 attention；波形与耦合分成两组 token。'),
 ('Duplex-F192','论文双路版','cf2_v1d192','PACLock','8 频段，D192，关闭独立频段 mixer，将频段并入空间 attention（space_over_bands）。论文当前参考模型。'),
 ('PAC8','8 频段纯耦合版','paclock_rot2','PACLock','8 频段，D128，幅度乘以 PAC 对齐的单位相位；没有独立波形 token。'),
 ('PAC16','16 频段纯耦合版','crofremo_n5','PACLock','16 频段，D128，纯耦合；n5 配方：wd .05、dropout .35、五种增强。'),
 ('Quad16','正交载波版','nb16_quadrature_20260915','CFM4EEG-nbands-20260915','16 频段，D128，幅度和波形占复数的两个坐标，再作 PAC 旋转；沿用 n5 配方。'),
 ('Carrier16','波形载波版','nb16_carrier_20260915','CFM4EEG-nbands-20260915','16 频段，D128，复数两个坐标都由波形投影得到，再作 PAC 旋转；沿用 n5 配方。'),
 ('Anchor16','有界调制版','nb16_anchored_20260915','CFM4EEG-anchored-watch-20260915','16 频段，D128，波形载波从恒等映射起步，学习有界 PAC 旋转；沿用 n5 配方。'),
 ('Residual16','波形残差版','nb16_residual_20260915','CFM4EEG-anchored-watch-20260915','16 频段，D128，纯耦合 token 加可学习的频段内波形残差；沿用 n5 配方。'),
]
def read(p):return json.loads(p.read_text()) if p.exists() else None
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
records=[]
for alias,label,suffix,checkout,description in VERSIONS:
 v=dict(alias=alias,label=label,run_suffix=suffix,checkout=str(BASE/checkout),description=description,datasets={})
 for ds in ['tuev','chbmit']:
  root=BASE/checkout/'runs'/(ds+'-'+suffix);rows=[]
  for p in sorted(root.glob('seed*/result.json')):
   r=read(p);rows.append(dict(seed=r['seed'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),test=r['test'],best_val=r.get('best_val'),selected_validation=r.get('selected_validation'),stopped_by=r.get('stopped_by'),config=r.get('config'),n_params_M=r.get('n_params_M')))
  live=[]
  for p in sorted(root.glob('seed*/progress.json')):
   if (p.parent/'result.json').exists():continue
   r=read(p);h=r.get('history',[]);live.append(dict(path=str(p),last=h[-1:] ,pid=r.get('pid'),note='Progress snapshot; process liveness verified separately in allocation 419232.'))
  v['datasets'][ds]=dict(completed=rows,incomplete=live)
 records.append(v)
registry=dict(snapshot_utc=now,versions=records,rule='Alias is human-facing only. Never rename original runs, checkpoints, or paper table keys. Test summaries describe existing results, not model-selection gates.')
(OUT/'docs/MODEL-VERSIONS-2026-09-15.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
lines=['# CroFreMo 模型版本索引','','快照 UTC：'+now,'','## 使用规则','','- 今后明确使用本索引简称，不再单独说“duplex”。简称不表示性能排序。','- 原始配置、run 目录和 checkpoint 名不改；每个简称映射固定的原始 ID。','- 结果必须附数据集、指标、验证/测试、seed 数和结束原因；未完成不填测试分数。','- 新结构先登记结构差异、对照、验证门槛、预算、配置 ID，再提交。完成后保留失败与预算截断记录。','- 下表是两套数据上的版本导航，不是整个项目所有历史实验，也不是 SOTA 排名。','- 同一简称允许数据集必要的损失/类别数差异；完整逐 seed 配置见伴随 JSON。','','## 结构和原始 ID','', '| 简称 | 中文名 | 原始 run 后缀 | 结构 |','|---|---|---|---|']
for v in records:lines.append('| '+ ' | '.join([v['alias'],v['label'],'`'+v['run_suffix']+'`',v['description']])+' |')
lines+=['','## 已完成测试结果','','均值 ± 样本标准差；单 seed 不写 ±。TUEV 用 κ，CHBMIT 用 PR-AUC。','', '| 简称 | TUEV κ | CHBMIT PR-AUC |','|---|---:|---:|']
for v in records:
 cells=[]
 for ds,key in [('tuev','cohen_kappa'),('chbmit','pr_auc')]:
  d=v['datasets'][ds];vals=[r['test'][key] for r in d['completed']]
  if len(vals)>1:s=f'{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}（{len(vals)} seeds）'
  elif vals:s=f'{vals[0]:.4f}（1 seed）'
  elif d['incomplete']:s='训练中，暂无完整测试结果'
  elif v['alias']=='Residual16' and ds=='tuev':s='节点内补位队列，尚未开跑'
  else:s='本次核对未发现该配置完整结果'
  cells.append(s)
 lines.append('| '+v['alias']+' | '+' | '.join(cells)+' |')
lines+=['','## 当前判断与在跑任务','','- **Duplex-F192 是论文版**；Duplex-A128 是早期版。此前将 A128 的数字用来回答论文版比较，已纠正。','- Carrier16 两项主指标的 seed0 数字都高于论文版均值；不能据此宣称稳定领先。TUEV balanced accuracy 只有 .6064，不能说所有指标都强。','- 旧 CHB 验证筛选门槛参照 Duplex-A128，而非论文版。该历史门槛不因查看测试分数而倒改；未达门槛不等于模型没有研究价值。','- 当前 allocation 419232：Quad16/CHB、Anchor16/CHB、Anchor16/TUEV、Residual16/CHB。Residual16/TUEV 在下一张空卡冒烟后补位。','- 实时状态看 results/amd-handover-419232.json、results/backfill-419232.json 和训练 progress.json；本文件是带日期快照。','','## 溯源','','逐 seed 测试值、选中验证指标、完整配置、原始绝对路径及 SHA256：`MODEL-VERSIONS-2026-09-15.json`。','论文映射：`scripts/gen_tables.py` 的主表 CroFreMo → `cf2_v1d192`。不手改生成表格。','']
(OUT/'docs/MODEL-VERSIONS.md').write_text('\n'.join(lines))
print('\n'.join(lines[lines.index('## 已完成测试结果'):]))
