"""Fail before allocating expensive training if data/review/smoke are not ready."""
import argparse,hashlib,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import yaml

a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--mode',choices=['smoke','train'],required=True);args=a.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text())
root=Path(os.path.expandvars(cfg['data_root']))
ready=root/'ACCEPTED.json'
if not ready.exists():raise SystemExit(f'Dataset not accepted: {ready}')
receipt=json.loads(ready.read_text())
if receipt.get('status')!='accepted':raise SystemExit('Dataset acceptance failed')
manifest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
if receipt.get('manifest_sha256')!=manifest:raise SystemExit('Dataset manifest changed after acceptance')
if cfg.get('require_rebuild_ready'):
 rebuild=json.loads((root/'READY.json').read_text())
 if rebuild.get('manifest_sha256')!=manifest:raise SystemExit('Rebuilt dataset not READY for accepted manifest')
expected={f'{split}_{kind}.npy' for split in ('train','val','test') for kind in ('signals','labels')}
if not expected.issubset(receipt.get('files',{})):raise SystemExit('Incomplete acceptance file inventory')
for name,stat in receipt.get('files',{}).items():
 p=root/name
 if p.stat().st_size!=stat['size']:raise SystemExit(f'Dataset size changed: {p}')
 if 'mtime_ns' in stat and p.stat().st_mtime_ns!=stat['mtime_ns']:raise SystemExit(f'Dataset timestamp changed: {p}')
review_path=Path('review/accepted_configs.json')
if not review_path.exists():raise SystemExit('No reviewed-config receipt')
review=json.loads(review_path.read_text())
key=str(Path(args.config))
if review.get(key)!=hashlib.sha256(Path(args.config).read_bytes()).hexdigest():raise SystemExit(f'Configuration not reviewed: {key}')
if cfg.get('disabled_reason'):raise SystemExit(cfg['disabled_reason'])
if args.mode=='train':
 smoke=Path(os.environ.get('CFM_REBUILD_ROOT','/scratch/zz5070/CFM4EEG-rebuild-20261021'))/'smoke'/cfg['name']/'seed0'/'smoke.json'
 if not smoke.exists():raise SystemExit(f'Missing smoke pass: {smoke}')
 data=json.loads(smoke.read_text())
 if data.get('status')!='passed' or data.get('config')!=cfg:raise SystemExit('Smoke/config mismatch')
 if data.get('checkpoint_reload')!='passed':raise SystemExit('Checkpoint reload not verified')
 from paclock_bench.training.recipe_runtime import provenance
 now=provenance(args.config,cfg,{})
 before=data.get('provenance',{})
 if now['source_sha256']!=before.get('source_sha256'):raise SystemExit('Code changed since smoke')
 if now.get('vendor_source_sha256')!=before.get('vendor_source_sha256'):raise SystemExit('Upstream source changed since smoke')
 if cfg.get('runner') and now.get('runtime_source_sha256')!=before.get('runtime_source_sha256'):raise SystemExit('Native runner changed since smoke')
 if now.get('pretrained_payloads')!=before.get('pretrained_payloads'):raise SystemExit('Pretrained weights changed since smoke')
 if before.get('input_contract',{}).get('manifest_sha256')!=manifest:raise SystemExit('Data changed since smoke')
print(f'PREFLIGHT PASS dataset={cfg["dataset"]} model={cfg["model"]} mode={args.mode}',flush=True)
