"""Stage-specific TFM gate/orchestration for separate2/2/1-GPU Slurm jobs.

--plan prints a dependency graph. --mode auto smokes and executes ONE stage;
Slurm should allocate2 GPUs for tokenizer/MTP and1 for fine-tuning. Exit75
means wall-clock interruption with full Lightning recovery, never completion.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def use_local_lightning_launcher():
    """The chain owns one Slurm task; Lightning creates the local GPU workers.

    Inherited task metadata would make Lightning assume Slurm already launched
    every DDP rank. Keep job/allocation identity for the outer requeue wrapper.
    """
    keys = ('SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE', 'SLURM_PROCID',
            'SLURM_LOCALID', 'SLURM_NODEID')
    removed = {key: os.environ.pop(key) for key in keys if key in os.environ}
    if removed:
        print(json.dumps({'launcher': 'lightning_local_ddp',
                          'removed_slurm_task_metadata': removed,
                          'slurm_job_id': os.environ.get('SLURM_JOB_ID')}), flush=True)
    return removed


def make_plan(config,seed,root):
    setting=config.get('tfm_setting','single')
    def expand(value):
        value=os.path.expandvars(os.path.expanduser(value))
        return Path(value) if '$' in value else Path(value).resolve()
    result=[]
    if setting=='single':
        tokenizer=expand(config['tokenizer_path']);encoder=expand(config['encoder_path'])
        result += [dict(stage='tokenizer',gpus=2,depends_on=None,out=str(tokenizer.parent),artifact=str(tokenizer),epochs=100),
                   dict(stage='mtp',gpus=2,depends_on='tokenizer',out=str(encoder.parent),artifact=str(encoder),epochs=50)]
    result.append(dict(stage='finetune',gpus=1,depends_on='mtp' if setting=='single' else None,
                       out=str(root/'runs'/config['name']/f'seed{seed}'),epochs=50))
    return result


def main():
    use_local_lightning_launcher()
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--stage',choices=['tokenizer','mtp','finetune']);p.add_argument('--mode',choices=['smoke','train','auto'],default='auto')
    p.add_argument('--plan',action='store_true');p.add_argument('--smoke-steps',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=0);p.add_argument('--resume',default='auto')
    a=p.parse_args();os.chdir(ROOT)
    config_path=Path(a.config).resolve();config_key=str(config_path.relative_to(ROOT))
    cfg=yaml.safe_load(config_path.read_text());root=Path(os.environ['CFM_REBUILD_ROOT']).resolve()
    os.environ.setdefault('PACLOCK_CKPT',str(root/'ckpt'))
    plan=make_plan(cfg,a.seed,root)
    if a.plan:
        print(json.dumps({'config':config_key,'stages':plan,'resume_rule':'Exit75 requires resuming same stage; only completed.json permits dependents.'},indent=2));return
    if a.stage is None:raise ValueError('--stage is required unless --plan')
    stage=next((s for s in plan if s['stage']==a.stage),None)
    if stage is None:raise ValueError('Released multiple setting runs fine-tuning only; no fake target-only pretraining')
    subprocess.run([sys.executable,str(ROOT/'scripts/rebuild/preflight.py'),'--config',config_key,'--mode','smoke'],check=True)
    from paclock_bench.training.recipe_runtime import provenance
    from scripts.rebuild.native_adaptations import tfm_stage_input_config
    data_root=Path(os.path.expandvars(cfg['data_root']));manifest=digest(data_root/'manifest.json')
    input_cfg=tfm_stage_input_config(cfg,a.stage)
    current=provenance(config_key,input_cfg,{'input_contract':{'manifest_sha256':manifest}})
    binding_keys=['source_sha256','runtime_source_sha256','vendor_source_sha256','pretrained_payloads']
    def validate_binding(before):
        for key in binding_keys:
            if before.get(key)!=current.get(key):raise RuntimeError(f'{a.stage}: {key} changed; new stage smoke required')
        if before.get('input_contract',{}).get('manifest_sha256')!=manifest:raise RuntimeError('Dataset changed')
    def completed(item):
        cp=Path(item['out'])/'completed.json'
        if not cp.exists():return False
        c=json.loads(cp.read_text())
        if c.get('stage')!=item['stage'] or c.get('dataset')!=cfg['dataset'] or c.get('manifest_sha256')!=manifest:
            raise RuntimeError('TFM dependency completion dataset/stage mismatch')
        if c.get('config_sha256')!=digest(config_path):raise RuntimeError('TFM dependency trained under another config')
        expected_seed=a.seed if item['stage']=='finetune' else cfg.get('stage_seed',5)
        if c.get('seed')!=expected_seed:raise RuntimeError('TFM stage seed mismatch')
        if c.get('epochs',0)<item['epochs'] or c.get('actual_steps',0)<c.get('expected_steps',1):
            raise RuntimeError('TFM dependency was incomplete')
        if digest(c['export'])!=c.get('export_sha256'):raise RuntimeError('TFM dependency artifact hash mismatch')
        before=c.get('provenance',{})
        for key in ['source_sha256','runtime_source_sha256','vendor_source_sha256']:
            if before.get(key)!=current.get(key):raise RuntimeError('TFM completed stage source binding mismatch: '+key)
        expected_inputs=provenance(config_key,tfm_stage_input_config(cfg,item['stage']),{})['pretrained_payloads']
        if before.get('pretrained_payloads')!=expected_inputs:raise RuntimeError('TFM completed stage initialization weights changed')
        return True
    for item in plan:
        if item['stage']==a.stage:break
        if not completed(item):raise RuntimeError(f'Finish train-only {item["stage"]} before {a.stage}')
    out=Path(stage['out']);out.mkdir(parents=True,exist_ok=True)
    with (out/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if completed(stage):
            # Checkpoint reuse is explicit; no training/output overwrite.
            print(json.dumps({'status':'already_completed','stage':a.stage,'out':str(out)}));return
        smoke_seed=a.seed if a.stage=='finetune' else cfg.get('stage_seed',5)
        smoke_root=root/'smoke'/cfg['name']/f'seed{smoke_seed}'/'tfm'/a.stage
        receipt=smoke_root/'smoke.json'
        runner=ROOT/'scripts/rebuild/run_tfm_stage.py'
        base=[sys.executable,'-u',str(runner),config_key,'--stage',a.stage,'--seed',str(a.seed)]
        if a.mode in ('smoke','auto') and not receipt.exists():
            if smoke_root.exists() and any(smoke_root.iterdir()):
                # Preserve failed smoke artifacts and make a fresh attempt.
                import time
                smoke_root.rename(smoke_root.with_name(smoke_root.name+f'.failed-{time.time_ns()}'))
            subprocess.run(base+['--out',str(smoke_root),'--smoke-steps',str(a.smoke_steps)],check=True)
        if not receipt.exists():raise RuntimeError(f'Missing stage-specific smoke: {receipt}')
        smoke=json.loads(receipt.read_text())
        if (smoke.get('status')!='passed' or smoke.get('config')!=cfg or smoke.get('stage')!=a.stage
            or smoke.get('checkpoint_reload')!='passed' or smoke.get('test_evaluated') is not False):
            raise RuntimeError('Stage smoke identity/checkpoint/test-access check failed')
        validate_binding(smoke.get('provenance',{}))
        if a.mode=='smoke':return
        command=base+['--out',str(out)]
        checkpoint=out/'last.ckpt'
        if a.resume=='auto' and checkpoint.exists():command+=['--resume',str(checkpoint)]
        elif a.resume not in ('auto','none'):command+=['--resume',a.resume]
        elif checkpoint.exists():raise RuntimeError('Existing partial stage requires resume')
        if a.max_hours:command+=['--max-hours',str(a.max_hours)]
        result=subprocess.run(command)
        if result.returncode==75:
            print(json.dumps({'status':'paused','stage':a.stage,'resume':str(checkpoint)}));raise SystemExit(75)
        result.check_returncode()
        if not completed(stage):raise RuntimeError('Native stage exited without complete artifact')
        print(json.dumps({'status':'completed','stage':a.stage,'out':str(out)}))


if __name__=='__main__':main()
