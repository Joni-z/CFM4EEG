"""Known-PAC measurement assay for the unchanged, original CfFT checkpoint.

Runs only Sinc -> Hilbert -> native patch_pac_vector. No token projection,
rotation, encoder, fitting, or gradients. Synthetic signals are reproducible;
all conditions and unsuccessful cases remain in the recording-level output.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy import signal, stats
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.analytic import hilbert, phase_amplitude
from paclock_bench.models.paclock.frontend.triaxial import NORM_CONST, patch_pac_vector

CHECKPOINT_SHA = '1fb2b132f935d890373f38f47ee2b5e99c796b3d5d9b76773c74b94b3e8b1a3c'
PAIRS = [('delta_beta', 2., 20.), ('theta_gamma', 6., 40.),
         ('alpha_gamma', 10., 60.), ('boundary_0p5', .5, 20.),
         ('boundary_1', 1., 20.)]
DEPTHS = [.2, .4, .6, .8]
PHASES = [0., math.pi / 2, math.pi, 3 * math.pi / 2]
NOISES = [('clean', None), ('10dB', 10.), ('0dB', 0.)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / 'paclock_bench/models').rglob('*.py'))}


def load_original(checkpoint):
    assert sha256(checkpoint) == CHECKPOINT_SHA, 'Not the declared original checkpoint'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], (16, 1000)).cpu()
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    stem = model.frontend
    assert stem.patch_len == stem.pac_patch_len == 50
    assert stem.pac_patch_lens == [50]
    assert stem.tokenizer_mode == 'pac_interaction'
    assert stem.pac_token_mode == 'measured'
    assert stem.interaction_mode == 'rotation'
    return model, saved['config']


def band_mapping(stem):
    kernels = stem.sinc.filters().detach().cpu().numpy()[:, 0]
    tap = np.arange(kernels.shape[-1]) - kernels.shape[-1] // 2
    grid = np.linspace(0, 100, 2001)
    spectrum = abs(kernels @ np.exp(2j * np.pi * tap[:, None] * grid[None] / 200))
    maps = {}
    for name, low, high in PAIRS:
        hz = np.array([low, high, high - low, high + low])
        response = kernels @ np.exp(2j * np.pi * tap[:, None] * hz[None] / 200)
        src, dst = int(abs(response[:, 0]).argmax()), int(abs(response[:, 1]).argmax())
        assert src < dst
        source_mask = abs(response[:, 0]) >= .5 * abs(response[:, 0]).max()
        target_mask = abs(response[:, 1]) >= .5 * abs(response[:, 1]).max()
        region = source_mask[:, None] & target_mask[None]
        region &= np.triu(np.ones(region.shape, bool), 1)
        maps[name] = dict(source=src, target=dst, accepted_edges=np.argwhere(region).tolist(),
                          chance_top1=float(region.sum() / (16 * 15 / 2)),
                          phase_reference_offset=float(np.angle(response[src, 0])),
                          carrier_gain=float(abs(response[dst, 1])),
                          sideband_relative_gain=(abs(response[dst, 2:]) / abs(response[dst, 1])).tolist(),
                          source_hz=low, target_hz=high)
    return maps, grid, spectrum


def make_signal(spec, base_seed):
    # Same realization across coupling depths, phases, and noise levels.
    # Calibration and held-out nulls have disjoint random streams.
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, spec['pair_index'],
                                                       spec['trial'], spec['stream']]))
    t = np.arange(1000) / 200
    low = spec['source_hz'] * rng.uniform(.97, 1.03)
    high = spec['target_hz'] * rng.uniform(.985, 1.015)
    phase = 2 * np.pi * low * t + rng.uniform(-np.pi, np.pi)
    carrier = 2 * np.pi * high * t + rng.uniform(-np.pi, np.pi)
    depth = spec['depth']
    if spec['kind'] == 'independent':
        # Independent smooth envelope with the same arcsine marginal as cos(phi).
        # A fixed time shift of a periodic envelope would NOT remove PAC.
        ext = rng.normal(size=4000)
        freq = np.fft.rfftfreq(ext.size, 1 / 200)
        weight = np.exp(-.5 * ((freq - low) / max(.4 * low, .25)) ** 2)
        smooth = np.fft.irfft(np.fft.rfft(ext) * weight, n=ext.size)[1500:2500]
        rank = stats.rankdata(smooth, method='ordinal')
        fluctuation = np.sin(np.pi * ((rank - .5) / t.size - .5))
    else:
        fluctuation = np.cos(phase - spec['preferred_phase'])
    envelope = 1 + depth * fluctuation
    fast = envelope * np.cos(carrier)
    scale = math.sqrt(.5 / np.mean(fast ** 2))
    fast *= scale
    envelope *= scale
    slow = .8 * np.cos(phase)
    distractor = .15 * np.cos(2 * np.pi * 13 * t + rng.uniform(-np.pi, np.pi))
    distractor += .15 * np.cos(2 * np.pi * 28 * t + rng.uniform(-np.pi, np.pi))
    clean = slow + fast + distractor
    noise = rng.normal(size=t.size)
    freq = np.fft.rfftfreq(t.size, 1 / 200)
    noise = np.fft.irfft(np.fft.rfft(noise) / np.sqrt(np.maximum(freq, .2)), n=t.size)
    noise -= noise.mean()
    if spec['noise_db'] is not None:
        noise *= np.sqrt(np.mean(clean ** 2) / 10 ** (spec['noise_db'] / 10)) / noise.std()
        raw = clean + noise
    else:
        raw = clean
    return raw.astype(np.float32), np.exp(1j * phase).astype(np.complex64), envelope.astype(np.float32), {
        'actual_source_hz': low, 'actual_target_hz': high,
        'fast_rms': float(np.sqrt(np.mean(fast ** 2))),
        'envelope_mean': float(envelope.mean()), 'clean_rms': float(np.sqrt(np.mean(clean ** 2)))}


def cases(trials, calibration):
    for pair_index, (name, low, high) in enumerate(PAIRS):
        for noise_name, noise_db in NOISES:
            common = dict(pair=name, pair_index=pair_index, source_hz=low,
                          target_hz=high, noise=noise_name, noise_db=noise_db)
            for trial in range(trials):
                yield {**common, 'kind':'constant_null', 'split':'test', 'trial':trial,
                       'depth':0., 'preferred_phase':0., 'stream':0}
                for depth in DEPTHS:
                    for theta in PHASES:
                        yield {**common, 'kind':'coupled', 'split':'test', 'trial':trial,
                               'depth':depth, 'preferred_phase':theta, 'stream':0}
                    yield {**common, 'kind':'independent', 'split':'test', 'trial':trial,
                           'depth':depth, 'preferred_phase':0., 'stream':1}
            for trial in range(calibration):
                yield {**common, 'kind':'independent', 'split':'calibration', 'trial':trial,
                       'depth':DEPTHS[trial % len(DEPTHS)], 'preferred_phase':0., 'stream':2}


@torch.inference_mode()
def measurement(stem, x):
    b, c, length = x.shape
    filtered = stem.sinc(x.reshape(b * c, 1, length)).reshape(b, c, stem.n_bands, length)
    phase, amplitude = phase_amplitude(hilbert(filtered))
    z = patch_pac_vector(phase, amplitude, length // stem.pac_patch_len, stem.normalize)
    # Offline signal-generator sanity reference, never fed to the model.
    # Added after the initial native assay, and declared as a diagnostic only.
    centered = amplitude[:, 0] - amplitude[:, 0].mean(-1, keepdim=True)
    reference = torch.einsum('bit,bjt->bij', phase[:, 0], centered.to(phase.dtype)) / length
    return (z[:, 0].numpy(),
            amplitude[:, 0].reshape(b, stem.n_bands, 20, 50).mean(-1).numpy(),
            reference.numpy())


def wrap(angle):
    return np.angle(np.exp(1j * angle))


@torch.inference_mode()
def smoke_check(model, stem, x):
    direct, _, _ = measurement(stem, x)
    captured = {}
    class StopBeforeTokens(Exception):
        pass
    def capture(ph, amp, vectors, **kwargs):
        captured['z'] = vectors[0][:, 0].detach().numpy()
        raise StopBeforeTokens()
    # This observation-only intercept halts native forward before token projection.
    with patch.object(stem, '_interaction_tokens', side_effect=capture):
        try:
            stem(x)
        except StopBeforeTokens:
            pass
    np.testing.assert_array_equal(direct, captured['z'])
    raw = x[:, 0].numpy()
    kernels = stem.sinc.filters().detach().numpy()[:, 0]
    independent_filtered = np.stack([[np.convolve(v, k, mode='same') for k in kernels] for v in raw])
    analytic = signal.hilbert(independent_filtered, axis=-1)
    amplitude = abs(analytic)
    phase = analytic / np.maximum(amplitude, 1e-6)
    ph = phase.reshape(len(raw), 16, 20, 50)
    am = amplitude.reshape(len(raw), 16, 20, 50)
    am = am - am.mean(-1, keepdims=True)
    independent_z = np.einsum('bipl,bjpl->bpij', ph, am) / 50 / NORM_CONST
    np.testing.assert_allclose(direct, independent_z, rtol=4e-4, atol=3e-6)
    time_axis=torch.arange(1000)/200
    unit=torch.exp(2j*torch.pi*2*time_axis)
    ph=torch.stack([unit,unit])[None,None]
    constant=torch.ones(1,1,2,1000)
    assert torch.count_nonzero(patch_pac_vector(ph,constant,20,True))==0
    am=constant.clone()
    am[:,:,1]=1+.6*torch.cos(2*torch.pi*2*time_axis-1.1)
    known=patch_pac_vector(ph,am,1,True)[0,0,0,0,1].numpy()*NORM_CONST
    np.testing.assert_allclose(known,.3*np.exp(1.1j),atol=1e-6,rtol=1e-6)
    return {'native_forward_exact_match':True,
            'independent_numpy_scipy_max_Z_error':float(abs(direct - independent_z).max()),
            'constant_envelope_exact_zero':True,'known_population_vector_sign_and_scale':True,
            'token_projection_rotation_encoder_executed':False}


def summarize(rows, seed):
    rng = np.random.default_rng(seed)
    summary = []
    for name, _, _ in PAIRS:
        for noise, _ in NOISES:
            subset = [r for r in rows if r['pair'] == name and r['noise'] == noise]
            cal = [r for r in subset if r['split'] == 'calibration']
            null = [r for r in subset if r['split'] == 'test' and r['kind'] == 'independent']
            pos = [r for r in subset if r['kind'] == 'coupled']
            for scope in ['all', 'interior']:
                field = 'depth_' + scope
                threshold = float(np.quantile([r[field] for r in cal], .95))
                false = np.array([r[field] > threshold for r in null])
                null_trials = sorted({r['trial'] for r in null})
                null_trial_rates = np.array([np.mean([r[field] > threshold for r in null if r['trial']==t]) for t in null_trials])
                null_boot = rng.choice(null_trial_rates,size=(2000,len(null_trials)),replace=True).mean(1)
                phase_errors = np.array([r['phase_error_' + scope] for r in pos])
                # Cluster bootstrap: depths and phases share each realization.
                trial_ids = sorted({r['trial'] for r in pos})
                trial_mae = np.array([np.mean([r['phase_error_' + scope] for r in pos if r['trial'] == t]) for t in trial_ids])
                boot = rng.choice(trial_mae, size=(2000, len(trial_mae)), replace=True).mean(1)
                trial_loc = np.array([np.mean([r['localized_' + scope] for r in pos if r['trial']==t]) for t in trial_ids])
                loc_boot = rng.choice(trial_loc,size=(2000,len(trial_loc)),replace=True).mean(1)
                rho = stats.spearmanr([r['depth'] for r in pos], [r[field] for r in pos]).statistic
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score([1]*len(pos) + [0]*len(null), [r[field] for r in pos + null])
                by_depth = {str(d):float(np.mean([r[field] for r in pos if r['depth'] == d])) for d in DEPTHS}
                summary.append(dict(pair=name, noise=noise, scope=scope, coupled=len(pos),
                    independent_null=len(null), calibration_null=len(cal), realizations=len(trial_ids),
                    localization_rate=float(np.mean([r['localized_' + scope] for r in pos])),
                    localization_ci95=np.quantile(loc_boot,[.025,.975]).tolist(),
                    strength_spearman=float(rho), phase_mae_deg=float(phase_errors.mean()),
                    phase_mae_ci95=np.quantile(boot,[.025,.975]).tolist(),
                    recording_phase_mae_deg=float(np.mean([r['pooled_phase_error_' + scope] for r in pos])),
                    complex_depth_vector_mae=float(np.mean([r['vector_error_' + scope] for r in pos])),
                    mean_estimated_depth=by_depth, detection_auroc=float(auc),
                    null_threshold=threshold, false_positive_rate=float(false.mean()),
                    false_positive_ci95=np.quantile(null_boot,[.025,.975]).tolist(),
                    false_positive_independent_realizations=len(null_trials),
                    true_positive_rate=float(np.mean([r[field] > threshold for r in pos])),
                    oracle_phase_mae_deg=float(np.mean([r['oracle_phase_error_' + scope] for r in pos])),
                    oracle_depth_mae=float(np.mean([abs(r['oracle_depth_' + scope]-r['depth']) for r in pos])),
                    reference_recording_dpac_auroc=float(roc_auc_score([1]*len(pos)+[0]*len(null),[r['reference_depth'] for r in pos+null])),
                    reference_recording_dpac_phase_mae_deg=float(np.mean([r['reference_phase_error'] for r in pos]))))
    return summary


def figures(rows, summary, maps, grid, spectrum, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
    colors = {'clean':'#386A9C','10dB':'#D18B47','0dB':'#8F6B9D'}
    for j, (name, _, _) in enumerate(PAIRS[:3]):
        for noise, _ in NOISES:
            ss = next(s for s in summary if s['pair']==name and s['noise']==noise and s['scope']=='all')
            axes[0,j].plot(DEPTHS, [ss['mean_estimated_depth'][str(d)] for d in DEPTHS], 'o-', color=colors[noise], label=noise)
            axes[1,j].bar(['clean','10dB','0dB'].index(noise), ss['phase_mae_deg'], width=.6, color=colors[noise])
        axes[0,j].plot([0,.8],[0,.8],'--',color='.6',lw=1,label='ideal depth')
        axes[0,j].set(title=name.replace('_','–'),xlabel='Injected modulation depth',ylabel='Measured normalized depth')
        axes[1,j].axhline(90,ls='--',color='.6',lw=1)
        axes[1,j].set(xticks=[0,1,2],xticklabels=['clean','10 dB','0 dB'],ylabel='Patch preferred-phase error (°)',ylim=(0,180))
    axes[0,0].legend(fontsize=8)
    fig.savefig(output/'measurement-main.png',dpi=180)
    fig.savefig(output/'measurement-main.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11,3.5), constrained_layout=True)
    for j, metric in enumerate(['localization_rate','detection_auroc','false_positive_rate']):
        for noise, _ in NOISES:
            values=[next(s for s in summary if s['pair']==p and s['noise']==noise and s['scope']=='all')[metric] for p,_,_ in PAIRS]
            axes[j].plot(range(5),values,'o-',label=noise,color=colors[noise])
        axes[j].set(xticks=range(5),xticklabels=['δ–β','θ–γ','α–γ','0.5 Hz','1 Hz'],title=metric.replace('_',' '),ylim=(0,1.05))
    axes[0].legend(fontsize=8)
    fig.savefig(output/'measurement-controls.png',dpi=180)
    fig.savefig(output/'measurement-controls.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,3.4),constrained_layout=True)
    ax.plot(grid,(spectrum / spectrum.max(1,keepdims=True)).T,lw=.9,alpha=.8)
    ax.set(xlim=(0,75),xlabel='Frequency (Hz)',ylabel='Response / each filter peak',title='Frozen original CfFT filter responses')
    fig.savefig(output/'frozen-filter-responses.png',dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--calibration',type=int,default=200)
    parser.add_argument('--seed',type=int,default=18112026)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    model, config = load_original(args.checkpoint)
    stem=model.frontend
    initial_state=state_hash(model)
    sources=source_hashes()
    old=json.loads((ROOT/'experiments/audits/pac-token-readout-twoaxis-20261113.json').read_text())['rows']['measured']['source_hashes']
    for relative in ['triaxial.py','analytic.py','sinc.py']:
        assert sources['paclock_bench/models/paclock/frontend/'+relative]==old['paclock/frontend/'+relative]
    maps,grid,spectrum=band_mapping(stem)
    protocol=dict(study='original_cfft_known_pac_measurement_20261118',
        evaluation_level='measurement_before_token_construction',
        neural_training=False,probe_fitting=False,model_changed=False,device='cpu',
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=CHECKPOINT_SHA,
        config=config,source_sha256=sources,model_state_sha256_before=initial_state,
        harness_sha256=sha256(__file__),original_measurement_source_matches_20261113_audit=True,
        runtime=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=args.threads),
        generation=dict(fs=200,seconds=5,depths=DEPTHS,preferred_phases=PHASES,noises=NOISES,
                        trials=args.trials,calibration=args.calibration,seed=args.seed,
                        slow_frequency_jitter_fraction=.03,carrier_frequency_jitter_fraction=.015,
                        fast_component_rms=math.sqrt(.5),null='independent smooth envelope with matched arcsine marginal and fast RMS'),
        primary='Each original 0.25 s patch is scored; metrics averaged within recording before inference. All 20 patches retained.',
        secondary='Interior patches exclude 1 s at each boundary; pooled complex vector angle is reported separately and not substituted for patch accuracy.',
        post_initial_assay_reference='Offline 5 s global dPAC on the same filtered signals checks whether the synthetic positive/null are distinguishable; this is not an original-model result.',
        inference='95% CIs bootstrap the independent realization, clustering all phase/depth settings sharing its RNG seed. Not individual patches or phase/depth variants.',
        scale='Depth score = 2*abs(native Z)*fixed NORM_CONST / mean filtered target amplitude; evaluation normalization only; no fitted gain.',
        localization='Top native mean-absolute-Z upper-triangle edge; success inside response-derived half-maximum source/target region; no per-result selection.',
        null_threshold='95th percentile of separate independent-envelope calibration nulls, per frequency family and noise condition.',
        band_mapping=maps)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    specs=list(cases(args.trials,args.calibration))
    if args.smoke:
        specs=[s for s in specs if s['pair_index']==0][:32]
    total=len(specs)
    rows=[]
    started=time.monotonic()
    upper=np.triu(np.ones((16,16),bool),1)
    for first in range(0,total,32):
        batch=specs[first:first+32]
        generated=[make_signal(s,args.seed) for s in batch]
        x=torch.from_numpy(np.stack([g[0] for g in generated]))[:,None]
        if first==0:
            checks=smoke_check(model,stem,x[:2])
            (args.output/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
        z,amp,reference=measurement(stem,x)
        phase=np.stack([g[1] for g in generated])
        env=np.stack([g[2] for g in generated])
        oracle=patch_pac_vector(torch.from_numpy(np.stack([phase,phase],1))[:,None],
            torch.from_numpy(np.stack([np.ones_like(env),env],1))[:,None],20,True)[:,0,:,0,1].numpy()*NORM_CONST
        for n,spec in enumerate(batch):
            mapping=maps[spec['pair']]
            src,dst=mapping['source'],mapping['target']
            zp=z[n,:,src,dst]*NORM_CONST
            depth_vector=2*zp/np.maximum(amp[n,dst],1e-12)
            oracle_vector=2*oracle[n]/np.maximum(env[n].reshape(20,50).mean(-1),1e-12)
            truth_phase=spec['preferred_phase']+mapping['phase_reference_offset']
            reference_vector=2*reference[n,src,dst]/max(amp[n,dst].mean(),1e-12)
            coupled=spec['kind']=='coupled'
            row={**spec,**generated[n][3], 'source_band':src,'target_band':dst,
                 'reference_depth':float(abs(reference_vector)),
                 'reference_phase_error':float(abs(wrap(np.angle(reference_vector)-truth_phase))*180/np.pi) if coupled else None}
            for scope,select in [('all',slice(None)),('interior',slice(4,16))]:
                vector=depth_vector[select]
                phase_error=abs(wrap(np.angle(vector)-truth_phase))*180/np.pi
                score=abs(z[n,select]).mean(0)
                score=np.where(upper,score,-np.inf)
                edge=list(np.unravel_index(score.argmax(),score.shape))
                row.update({f'depth_{scope}':float(abs(vector).mean()),
                    f'phase_error_{scope}':float(phase_error.mean()) if coupled else None,
                    f'pooled_phase_error_{scope}':float(abs(wrap(np.angle(zp[select].mean())-truth_phase))*180/np.pi) if coupled else None,
                    f'vector_error_{scope}':float(abs(vector-(spec['depth'] if coupled else 0)*np.exp(1j*truth_phase)).mean()),
                    f'localized_{scope}':edge in mapping['accepted_edges'],
                    f'top_source_{scope}':int(edge[0]),f'top_target_{scope}':int(edge[1]),
                    f'oracle_depth_{scope}':float(abs(oracle_vector[select]).mean()),
                    f'oracle_phase_error_{scope}':float(np.mean(abs(wrap(np.angle(oracle_vector[select])-spec['preferred_phase'])))*180/np.pi) if coupled else None,
                    f'Z_real_{scope}':float(zp[select].real.mean()),f'Z_imag_{scope}':float(zp[select].imag.mean())})
            rows.append(row)
        if first==0 or (first//32)%20==0 or first+32>=total:
            elapsed=time.monotonic()-started
            progress=dict(done=len(rows),total=total,elapsed_seconds=elapsed,
                          estimated_remaining_seconds=elapsed*(total-len(rows))/len(rows))
            (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress),flush=True)
    with (args.output/'recordings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if not args.smoke:
        results=summarize(rows,args.seed+1)
        result=dict(study=protocol['study'],status='completed',evidence_type='synthetic_measurement',
            created_at=datetime.now().astimezone().isoformat(),neural_training=False,device='cpu',
            records=len(rows),wall_seconds=time.monotonic()-started,summary=results,
            checkpoint_sha256=CHECKPOINT_SHA,protocol_sha256=sha256(args.output/'protocol.json'),
            recordings_sha256=sha256(args.output/'recordings.csv'))
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        figures(rows,results,maps,grid,spectrum,args.output)
    assert initial_state==state_hash(model),'Model state changed'
    assert sources==source_hashes(),'Model source changed during measurement'
    assert sha256(args.checkpoint)==CHECKPOINT_SHA
    integrity={'checkpoint_unchanged':True,'model_state_unchanged':True,'model_source_unchanged':True,
               'rows':len(rows),'smoke_only':args.smoke,'native_forward_match':checks}
    (args.output/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print(json.dumps(integrity),flush=True)


if __name__=='__main__':
    main()
