"""Verify the handoff against recorded observations, without fitting new data."""
from pathlib import Path
import ast
import csv
import gzip
import hashlib
import json
import subprocess
import torch
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator
from paclock_bench.models.paclock.frontend.pac_calibration import PACCalibration, measurement_signature

root = Path(__file__).resolve().parent
repo = Path('/Users/mr.z/CFM4EEG-rebuild-20261021/repo')
def read(name):
    p = root/name
    if p.suffix == '.gz':
        with gzip.open(p, 'rt') as f:
            return json.load(f)
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

checks = {}
oracle = read('oracle-diagnosis.json')
checks['R1'] = oracle['signals'] == 960 and len(oracle['records']) == 3840
assert checks['R1']
reference = read('calibration-10s-fixed/records.json')
confirmation = read('calibration-confirm-10s/records.json')
op = CFCOperator()
counts = {}
for name, expected in [('marginal',199), ('tolerance95',161)]:
    cal = PACCalibration.load(root/f'calibration-final-meta-10s/{name}.json')
    refit = PACCalibration.fit([r['score'] for r in reference], op, 1, 2000,
                              alpha=cal.alpha, confidence=cal.confidence)
    assert refit == cal
    counts[name] = sum(r['score'] > cal.threshold for r in confirmation)
    assert counts[name] == expected
checks['R2'] = len(reference)==2000 and len(confirmation)==4000
assert not ({r['raw_sha256'] for r in reference} & {r['raw_sha256'] for r in confirmation})
ledger = list(csv.DictReader((root/'results.tsv').open(), delimiter='\t'))
assert len(ledger)==17 and {r['status'] for r in ledger} >= {'keep','discard','crash','confirm','diagnostic','baseline'}
for name in ['uncertain-smoke','baseline-dev','uncertain-dev','uncertain-b1-dev','uncertain-b4-dev']:
    manifest = read(name+'/manifest.json')
    assert manifest['driver_sha256'] == sha(root/'run_candidate_initial.py')
for name in ['uncertain-confirm','baseline-confirm','uncertain-grid-confirm']:
    assert read(name+'/manifest.json')['driver_sha256'] == sha(root/'frozen-dev/run_candidate.py')
checks['R3'] = True
raw = [r for r in read('baseline-confirm/records.json.gz') if r['variant']=='curve']
stable = [r for r in read('uncertain-confirm/records.json.gz') if r['variant']=='curve']
assert len(raw)==len(stable)==2688
assert all((a['seed'],a['raw_sha256'])==(b['seed'],b['raw_sha256']) for a,b in zip(raw,stable))
grid = read('uncertain-grid-confirm/records.json.gz')
assert len(grid)==2880 and any(not r['valid'] for r in grid)
assert read('carrier-confirm/result.json')['signals']==768
assert read('independence-audit.json')['duplicate_raw_signals']==0
assert read('independence-audit.json')['total_distinct']==10960
assert read('local-confirm-fresh/reproduction.json')['seed_override']==36000000
assert len(read('local-confirm-fresh/records.json.gz'))==128
assert all(read('frozen-algorithm-equivalence.json').values())
checks['R4'] = True
package = torch.load(root/'cfc-operator-ready.pt',weights_only=True,map_location='cpu')
restored = CFCOperator(**package['constructor'])
restored.load_state_dict(package['state_dict'])
assert all(torch.equal(restored.state_dict()[k],v) if isinstance(v,torch.Tensor)
           else restored.state_dict()[k]==v for k,v in op.state_dict().items())
assert sum(p.numel() for p in restored.parameters())==0
assert '44 passed' in (root/'final-tests.log').read_text()
checks['R5'] = True
cpu = read('engineering-cpu.json'); cuda = read('gpu-check.json')
assert cuda['parent_bfloat16_preserves_physical_state']
assert all(r['finite_features'] and r['finite_gradient'] and r['cpu_cuda_max_abs_error']<1e-5
           for r in cuda['records'].values())
assert read('checkpoint-cross-runtime.json')['loaded']
assert read('checkpoint-cross-runtime.json')['finite']
checks['R6'] = True
publication = read('publication-source.json')
actual = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',publication['main_commit']],cwd=repo,text=True).splitlines()
assert sorted(actual)==publication['changed_files']
assert not any('/tokenizer' in f or '/encoder' in f or f.endswith('train.py') for f in actual)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
checks['R7_local'] = True
out = dict(checks=checks, tests='44 passed', calibration_confirmation_counts=counts,
           source_commit=publication['main_commit'], development_commit=publication['development_commit'],
           scientific_execution_source=read('frozen-dev.json')['source_commit'] if 'source_commit' in read('frozen-dev.json') else read('confirmation-freeze.json')['source'],
           scope='Ready before tokenizer integration. No downstream training performed.',
           publication='Source and evidence tag remote verification is recorded separately.')
(root/'completion-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
