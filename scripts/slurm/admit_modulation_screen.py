"""Read-only CPU admission check for the four-run structural pilot."""
import argparse,hashlib,json,statistics,subprocess
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('--smoke-job',required=True);a=p.parse_args()
state=subprocess.check_output(['sacct','-n','-X','-j',a.smoke_job,'--format=JobIDRaw,State,ExitCode','-P'],text=True)
assert any(line.split('|')[:3]==[a.smoke_job,'COMPLETED','0:0'] for line in state.splitlines()),state
contract=json.loads(Path(f'results/modulation-contract-{a.smoke_job}.json').read_text())
assert {x['arm'] for x in contract}=={'quadrature','carrier'} and all(x['ok'] for x in contract)
records=[]
for stem in ['chbmit_quadrature','chbmit_carrier','tuev_quadrature','tuev_carrier']:
 cfg_path=Path(f'configs/modulation_carrier/{stem}_s0.yaml');cfg=yaml.safe_load(cfg_path.read_text())
 receipt=Path(f'results/modulation-smoke-{a.smoke_job}-{stem}.json');d=json.loads(receipt.read_text())
 assert d['ok'] and d['config_sha256']==hashlib.sha256(cfg_path.read_bytes()).hexdigest()
 assert d['smoke_sha256']==hashlib.sha256(Path('smoke/smoke_amd_partition.py').read_bytes()).hexdigest()
 assert d['warmup_excluded']>=2 and d['augmentation_path_warmups']
 assert d['peak_GiB']<60,d['peak_GiB']
 assert cfg['seed']==0 and cfg['epochs']==20 and cfg['batch_size']==32
 assert cfg['model_kwargs']['n_bands']==16 and cfg['model_kwargs']['waveform_modulation'] in ('quadrature','carrier')
 check_epochs=6 if cfg['dataset']=='chbmit' else 20
 projected=d['projected_train_seconds']*check_epochs/20
 assert projected < cfg['max_hours']*3600*.8,(stem,projected,cfg['max_hours'])
 out=Path('runs')/cfg['name']/'seed0'
 assert not out.exists(),f'Existing output {out}'
 records.append(dict(config=str(cfg_path),name=cfg['name'],seed=0,
   steady_step_seconds=statistics.mean(d['steady_step_seconds']),peak_GiB=d['peak_GiB'],
   full_20_epoch_train_hours=d['projected_train_seconds']/3600,
   admission_reference_epochs=check_epochs,reference_train_hours=projected/3600,
   max_hours=cfg['max_hours'],smoke_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
result=dict(ok=True,smoke_job=a.smoke_job,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   runtime_note='CHB is a bounded patience pilot; six observed reference epochs, not a full-schedule completion guarantee. Loading/evaluation add runtime.',runs=records)
p=Path('results/audits/modulation-carrier-admission-20260915.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
print(p.read_text())
