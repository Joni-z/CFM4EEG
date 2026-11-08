"""Read existing experiment artifacts into one static, source-linked inventory."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / '.local/workspace'
MAINT = BASE / 'maintenance'
TASK = MAINT / 'experiment-ledger-20261105'
OLD = PROJECT / '.local/legacy/experiments'
OUT = PROJECT / 'experiments'
PRIMARY = dict(tuev='cohen_kappa', iiic='cohen_kappa', chbmit='pr_auc',
               tusz='pr_auc', tuar='cohen_kappa', tuab='balanced_acc',
               sleepedf='cohen_kappa', isruc='cohen_kappa', adfd='balanced_acc',
               caueeg='balanced_acc', tuep='auroc')
PRUNE = {'source', 'release', 'clinical-release', 'vendor', 'venv', '__pycache__',
         '.git', 'env', 'macmini-client-env', 'site-packages', 'data', 'datasets'}
INTEREST = {'result.json', 'progress.json', 'stopped.json', 'provenance.json',
            'selection.json', 'config.json', 'summary.json', 'scores.json', 'results.tsv',
            'independent-check.json'}


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def result_key(value):
    # A later diagnostic verdict is not another optimizer execution.
    return digest({k:v for k,v in value.items() if k not in ('verdict','verdict_v1','validation_diagnostic')})


def scalar(value):
    if isinstance(value, (dict, list, tuple)):
        return compact(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return '' if value is None else value


def write_csv(name, rows, first=()):
    keys = list(dict.fromkeys([*first, *(k for row in rows for k in row)]))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: scalar(v) for k, v in row.items()} for row in rows)


def infer_dataset(document, cfg, path):
    for value in (document.get('dataset'), cfg.get('dataset'), cfg.get('task')):
        if isinstance(value, str):
            aliases = {'chbmit':'chbmit', 'chb_mit':'chbmit', 'chb-mit':'chbmit',
                       'tuev_shared16':'tuev', 'tuev_native23':'tuev', 'sleep-edf':'sleepedf',
                       'sleep_edf':'sleepedf', 'iiic':'iiic'}
            canonical = aliases.get(value.lower(), value.lower())
            return canonical, 'recorded' if canonical==value else 'recorded_alias:'+value
    match = re.search(r'(?:/|^)(chbmit|tuev|tusz|iiic|tuab|tuar|isruc|sleepedf|adfd|caueeg|tuep)(?:[-_/]|$)', path.lower())
    return (match.group(1), 'path') if match else ('', 'unknown')


def architecture(cfg, model, ref):
    mk = cfg.get('model_kwargs') or {}
    if not isinstance(mk, dict):
        mk = {}
    mode, mixer = mk.get('tokenizer_mode'), mk.get('freq_mixer')
    bands, width = mk.get('n_bands'), mk.get('d_model')
    folded = mk.get('space_over_bands')
    axes = '未记录'
    if mk.get('arch') == 'flat':
        axes = '平坦编码器'
    elif mixer == 'none' and folded is True:
        axes = '双轴：时间 / 电极×频带'
    elif mixer == 'none' and folded is False:
        axes = '时间 / 逐带电极；无跨带混合'
    elif mixer and mixer != 'none' and folded is False:
        axes = '三轴：时间 / 电极 / 频带'
    elif mixer and mixer != 'none' and folded is True:
        axes = '时间 / 电极×频带 / 频带混合'
    elif mixer and mixer != 'none':
        axes = '频带混合；空间折叠字段未记录'
    elif mk.get('arch') == 'triaxial':
        axes = '网格编码器；轴设置未记录'
    rows = bands * 2 if isinstance(bands, int) and mode in ('duplex', 'hybrid') else bands
    label, token = model or '模型未记录', mode or ''
    if model == 'wave_patch_twoaxis':
        label, token, axes, bands, rows = 'WavePatch 原波形双轴', '原波形线性投影', '双轴：时间 / 电极', 0, 1
    elif model == 'wave_patch_threeaxis':
        label, token, axes, bands, rows = 'WavePatch 学习内容行三轴', '原波形线性投影', '三轴：时间 / 电极 / 学习内容行', 0, mk.get('n_rows')
    elif model == 'paclock':
        token = {'raw': 'Sinc分频波形', 'pac_interaction': '关系变换的频带内容',
                 'duplex': '波形行＋耦合行', 'hybrid': '混合双路',
                 'cfc': 'CFC参考替换', 'cfc_direct': '单路径有界复增益',
                 'measured_pac': '显式PAC测量前端'}.get(mode, mode or '前端未记录')
        label = {'raw': '频带波形原始版', 'duplex': 'Duplex', 'pac_interaction': '频带关系内容版',
                 'cfc': 'CFC参考替换', 'cfc_direct': 'Candidate C 单路径',
                 'measured_pac': '显式PAC测量前端'}.get(mode, 'CfFT其他前端')
        if mode == 'duplex' and bands == 8 and width == 128 and mixer == 'attention':
            label = 'Duplex-A128'
        elif mode == 'duplex' and bands == 8 and width == 192 and mixer == 'none' and folded is True:
            label = 'Duplex-F192'
        elif mode == 'pac_interaction' and bands == 16 and mk.get('waveform_modulation') == 'quadrature':
            label = 'Quad-Range' if mk.get('filter_range') is not None else 'Quad16'
        if mode == 'cfc' and any(k in mk for k in ('cfc_guidance', 'cfc_stabilized', 'cfc_fusion')):
            label = '旧CFC集成（须绑定原源码）'
        suffix = []
        for k, text in [('pac_token_mode', ''), ('measurement_mode', '测量:')]:
            if mk.get(k) and mk[k] not in ('measured',):
                suffix.append(text + str(mk[k]))
        if mk.get('waveform_content') is False:
            suffix.append('envelope-only')
        if mk.get('waveform_modulation') in ('carrier', 'anchored'):
            suffix.append(mk['waveform_modulation'])
        if mk.get('local_residual'):
            suffix.append('local-residual')
        if mk.get('content_view'):
            suffix.append('content:'+str(mk['content_view']))
        if suffix:
            label += ' / ' + ', '.join(suffix)
    elif model and model.endswith(('_a128', '_quad16', '_quad_range')):
        host, family = model.split('_', 1)
        label = family + '前端 → ' + host
        bands = 8 if family == 'a128' else 16
        rows, token, axes = 16, family + '宿主前端', '宿主编码器'
    elif model and (model.endswith('_paclockfe') or model == 'biot_tfm_host'):
        label, axes = model + '宿主转接', '宿主编码器'
    # Reverse CBraMod transplantation had inactive inherited PAC config fields.
    if 'cbramod-tokenizer' in ref.lower() or 'cbramod_tokenizer' in ref.lower():
        label, token, axes, bands, rows = 'CBraMod前端 → CfFT双轴', 'CBraMod原生tokenizer＋适配', '双轴：时间 / 电极', 0, 1
    structural = {'model': model, 'kwargs': {k: v for k, v in mk.items() if k not in ('augmentations', 'dropout')},
                  'host_fields': {k: v for k, v in cfg.items() if k.startswith(('range_', 'adapter', 'band_mode', 'readout'))},
                  'effective_tokenizer': token}
    vid = digest(structural)[:16]
    return vid, dict(model_version=vid, model_display=label, model=model,
                     tokenizer=token, encoder_axes=axes, physical_bands=bands,
                     content_rows=rows, width=width, depth=mk.get('depth'),
                     patch_len=mk.get('patch_len'), pac_patch_len=mk.get('pac_patch_len'),
                     filter_range=mk.get('filter_range', cfg.get('range_filter_range')),
                     pac_mode=mk.get('pac_token_mode', cfg.get('range_pac_token_mode')),
                     waveform_modulation=mk.get('waveform_modulation'),
                     space_over_bands=folded, freq_mixer=mixer,
                     architecture_evidence='explicit_config' if folded is not None or model in ('wave_patch_twoaxis','wave_patch_threeaxis') else 'partial_config',
                     structure_json=structural)


def main():
    OUT.mkdir(exist_ok=True)
    sources, wrappers, input_hashes, errors = [], [], {}, []

    def add_source(host, path, document, raw_hash=None, mtime=None):
        if isinstance(document, dict):
            sources.append(dict(host=host, path=path, document=document,
                                sha256=raw_hash or digest(document), mtime=mtime))

    oldfile = OLD / 'raw-results-deduplicated.jsonl'
    input_hashes[str(oldfile)] = hashlib.sha256(oldfile.read_bytes()).hexdigest()
    for line in oldfile.read_text().splitlines():
        wrapped = json.loads(line)
        wrappers.append(wrapped)
        # Every historical result remains represented, including low scores and conflicts.
        add_source('archive', str(oldfile) + '#' + wrapped['result_id'], wrapped['result'])
    table_sources = []
    for directory, dirs, names in os.walk(MAINT):
        dirs[:] = [d for d in dirs if d not in PRUNE and not d.startswith(('source-', 'release-', 'runtime-'))
                   and not (Path(directory) == MAINT and d == TASK.name)]
        for name in names:
            if name not in INTEREST:
                continue
            p = Path(directory) / name
            if p.stat().st_size > 3 * 1024**2:
                continue
            raw = p.read_bytes()
            input_hashes[str(p)] = hashlib.sha256(raw).hexdigest()
            if name.endswith('.tsv'):
                table_sources.append(p)
                continue
            try:
                add_source('local', str(p), json.loads(raw), input_hashes[str(p)], p.stat().st_mtime)
            except (ValueError, UnicodeError) as error:
                errors.append({'path': str(p), 'error': str(error)})
    collections = [TASK/'collected/145-records.jsonl', *sorted((TASK/'collected').glob('145-records-*.jsonl'))]
    for collected in collections:
        input_hashes[str(collected)] = hashlib.sha256(collected.read_bytes()).hexdigest()
        for line in collected.read_text().splitlines():
            item = json.loads(line)
            if item['path'].endswith('.json'):
                assert hashlib.sha256(item['document'].encode()).hexdigest() == item['sha256']
                add_source(item['host'], item['path'], json.loads(item['document']), item['sha256'], item['mtime'])
    for receipt in sorted((OUT / 'receipts').glob('**/result.json')):
        raw = receipt.read_bytes()
        input_hashes[str(receipt)] = hashlib.sha256(raw).hexdigest()
        add_source('verified_receipt', str(receipt), json.loads(raw), input_hashes[str(receipt)])
    sidecars = {(s['host'], str(Path(s['path']).parent), Path(s['path']).name): s for s in sources}
    gpu_text = (TASK / 'collected/145-gpu-status.txt').read_text()
    live_pids = set(re.findall(r'^GPU-[^,]+,\s*(\d+),', gpu_text, flags=re.M))
    forced_stops = {str(s['document'].get('target_pid')):s for s in sources
                    if Path(s['path']).name=='forced-stop.json' and s['document'].get('target_pid')}
    artifact_sources = defaultdict(list)
    artifacts = {}
    old_ids = defaultdict(list)
    old_checkpoints = defaultdict(list)
    original_documents = {}
    for w in wrappers:
        h = result_key(w['result'])
        old_ids[h].append(w['result_id'])
        old_checkpoints[h].extend(w.get('checkpoints') or [])
        for s in w.get('sources', []):
            artifact_sources[h].append({k: s.get(k) for k in ('host', 'path', 'sha256', 'mtime')})
    for s in sources:
        d = s['document']
        if Path(s['path']).name != 'result.json' and s['host'] != 'archive':
            continue
        if not isinstance(d, dict):
            continue
        metrics = d.get('test') or d.get('test_metrics') or d.get('native_metrics') or d.get('metrics')
        if not isinstance(metrics, dict) or not metrics:
            continue
        h = result_key(d)
        artifacts[h] = s
        original_documents[digest(d)] = d
        artifact_sources[h].append({k: s.get(k) for k in ('host', 'path', 'sha256', 'mtime')})
    # Ignore duplicate progress documents once the exact same source directory has a result.
    finished_dirs = {(s['host'], str(Path(s['path']).parent)) for s in sources
                     if Path(s['path']).name == 'result.json' and result_key(s['document']) in artifacts}
    finished_configs = {digest([s['document']['config'],s['document'].get('seed',s['document']['config'].get('seed'))])
                        for s in sources if Path(s['path']).name=='result.json'
                        and isinstance(s['document'].get('config'),dict)
                        and not (s['document'].get('training_ongoing_at_snapshot') or s['document'].get('selection')
                                 or s['document'].get('evaluation_kind'))}
    progress_candidates = {}
    for s in sources:
        if Path(s['path']).name not in ('progress.json', 'stopped.json'):
            continue
        if (s['host'], str(Path(s['path']).parent)) in finished_dirs:
            continue
        d = s['document']
        if not isinstance(d.get('config'), dict) or not d.get('config', {}).get('dataset'):
            continue
        key = digest([d['config'],d.get('seed',d['config'].get('seed'))])
        if key in finished_configs and any(x in s['path'] for x in ('/checkpoint-archive/','/evaluations/','/stopped-for-identity/')):
            continue
        previous = progress_candidates.get(key)
        priority = (len(d.get('history') or []), s['host']=='145')
        if previous and priority <= previous[0]:
            continue
        progress_candidates[key] = (priority,s)
    for _, s in progress_candidates.values():
        d=s['document']
        h = result_key(d)
        if h not in artifacts:
            artifacts[h] = s
            original_documents[digest(d)] = d
            artifact_sources[h].append({k: s.get(k) for k in ('host', 'path', 'sha256', 'mtime')})

    rows, versions = [], {}
    for h, s in artifacts.items():
        d, path = s['document'], s['path']
        cfg = d.get('config') or {}
        if not isinstance(cfg, dict):
            cfg = {}
        selection = d.get('selection') or {}
        if not cfg and selection.get('checkpoint_source'):
            related = sidecars.get((s['host'], str(Path(selection['checkpoint_source']).parent), 'progress.json'))
            if related:
                cfg = related['document'].get('config') or {}
        dataset, dataset_from = infer_dataset(d, cfg, path)
        if not dataset:
            continue
        model = d.get('model') or cfg.get('model') or ''
        if s['host'] == 'verified_receipt':
            model = model.lower()
        vid, version = architecture(cfg, model, path)
        seed = d.get('seed', cfg.get('seed', selection.get('seed')))
        metrics = d.get('test') or d.get('test_metrics') or d.get('native_metrics') or d.get('metrics') or {}
        progress_only = Path(path).name in ('progress.json', 'stopped.json')
        snapshot = bool(d.get('training_ongoing_at_snapshot'))
        kind = 'clinical_checkpoint_eval' if selection or d.get('evaluation_kind') else 'clinical_run'
        if s['host'] == 'verified_receipt':
            kind = 'verified_completion_receipt'
        if snapshot:
            kind = 'interim_checkpoint_eval'
        if progress_only:
            kind = 'training_progress'
        stopped = d.get('stopped_by') or ''
        state = ('running' if s['host'] == '145' and str(d.get('pid')) in live_pids else 'incomplete_status_unknown') if progress_only else 'completed_record'
        if snapshot:
            state = 'interim_eval'
        elif stopped == 'time_budget':
            state = 'budget_truncated'
        elif stopped == 'operator_stop' or Path(path).name == 'stopped.json':
            state = 'operator_stopped'
        if progress_only:
            related_stop = sidecars.get((s['host'],str(Path(path).parent),'stopped.json'))
            if related_stop:
                state,stopped = 'operator_stopped','operator_stop'
            if str(d.get('pid')) in forced_stops:
                state,stopped = 'operator_stopped','operator_forced_stop'
        counts = d.get('class_counts') or {}
        cohort = d.get('cohort') or {}
        if not isinstance(cohort, dict):
            cohort = {}
        ntest = d.get('count') or d.get('test_rows') or d.get('rows')
        if not ntest and isinstance(counts.get('test'), list):
            ntest = sum(counts['test'])
        if not ntest:
            ntest = (cohort.get('split_windows') or {}).get('test')
        protocol = d.get('protocol') if isinstance(d.get('protocol'), str) else ''
        if dataset == 'chbmit' and ntest in (17386, 17224):
            protocol = 'legacy_enriched_17386' if ntest == 17386 else 'natural_grid_17224'
        elif dataset == 'tuev' and ntest == 29421:
            protocol = 'shared16_29421'
        elif dataset == 'tuev' and 'shared16' in str(d.get('dataset') or cfg.get('dataset')).lower():
            protocol = 'shared16_count_not_recorded'
        if not protocol:
            protocol = (cohort.get('input_scope') or 'protocol_not_bound')[:150]
        primary = d.get('primary_metric') or cfg.get('primary_metric') or PRIMARY.get(dataset, '')
        normalized = dict(metrics)
        for old, new in [('kappa', 'cohen_kappa'), ('balanced_accuracy', 'balanced_acc'),
                         ('roc_auc', 'auroc'), ('f1_weighted', 'weighted_f1')]:
            if new not in normalized and old in normalized:
                normalized[new] = normalized[old]
        # AP and trapezoidal PR-AUC remain distinct; native PR definitions need their source.
        metric_convention = 'project_trapezoid_PR' if model in ('paclock','wave_patch_twoaxis','wave_patch_threeaxis') else 'native_or_recorded_definition'
        checkpoint = d.get('selected_checkpoint') or d.get('checkpoint') or selection.get('checkpoint_source') or ''
        if isinstance(checkpoint, dict):
            checkpoint = compact(checkpoint)
        cksha = d.get('checkpoint_sha256') or d.get('selected_checkpoint_sha256') or selection.get('checkpoint_sha256') or ''
        if not checkpoint and old_checkpoints[h]:
            checkpoint = '; '.join(str(x.get('host', '')) + ':' + str(x.get('path', '')) for x in old_checkpoints[h])
        if not checkpoint and progress_only:
            checkpoint = str(Path(path).parent / 'best.pt') + ' (existence not inferred)'
        recipe = {k:v for k,v in cfg.items() if k not in ('name','group','data_root','seed','device','num_workers','reproduction')}
        data_identity = d.get('data_identity') or {}
        input_contract = d.get('input_contract') or {}
        identity = d.get('identity') or {}
        data_binding = (cohort.get('manifest_sha256') or input_contract.get('manifest_sha256')
                        or cfg.get('expected_manifest_sha256') or identity.get('manifest_sha256')
                        or (data_identity.get('manifest.json') or {}).get('sha256')
                        or d.get('data_manifest_sha256')
                        or d.get('data_manifest_created') or '')
        if dataset == 'chbmit' and protocol == 'protocol_not_bound':
            for receipt_path in [
                MAINT / 'cell-audit-tracker-20261022/chbmit-eegpt-approved.json',
                MAINT / 'status-later-20261026/chb-eegpt-pre/approval.json',
            ]:
                receipt = json.loads(receipt_path.read_text())
                if (cksha and receipt.get('checkpoint_sha256') == cksha) or (checkpoint and receipt.get('checkpoint') == checkpoint):
                    ntest = receipt.get('test_rows') or receipt.get('rows')
                    if ntest == 17224:
                        cksha = cksha or receipt.get('checkpoint_sha256', '')
                        protocol = 'natural_grid_17224'
                        data_binding = 'f914c074e9f7ab165fed5b27919cbbe9dbd12c5a3e0faf36756bb2bd87672567'
        setting = d.get('setting') or cfg.get('setting')
        if not setting and isinstance(cfg.get('pretrained'),bool):
            setting = 'pretrained' if cfg['pretrained'] else 'scratch'
        if not setting:
            setting = 'checkpoint_loaded' if cfg.get('checkpoint') else 'not_explicitly_recorded'
        history = d.get('history') or []
        verdict = d.get('verdict') or {}
        source_refs = sorted(set(str(x['host']) + ':' + str(x['path']) for x in artifact_sources[h]))
        row = dict(record_id=h[:20], evidence_type=kind, dataset=dataset,
                   model_display=version['model_display'], model_version=vid,
                   run_name=d.get('name') or cfg.get('name') or Path(path).parent.name,
                   seed=seed, status=state, protocol=protocol, primary_metric=primary,
                   primary_value=None if progress_only else normalized.get(primary),
                   **{k: None if progress_only else normalized.get(k) for k in ('cohen_kappa','pr_auc','average_precision','auroc','balanced_acc','weighted_f1','accuracy')},
                   encoder_axes=version['encoder_axes'], physical_bands=version['physical_bands'],
                   content_rows=version['content_rows'], n_params_M=d.get('n_params_M') or (d.get('num_parameters')/1e6 if d.get('num_parameters') else None),
                   setting=setting,
                   best_val=d.get('best_val', selection.get('best_val')), latest_tag=history[-1].get('tag') if history else selection.get('tag'),
                   peak_index=verdict.get('peak_index', d.get('best_epoch')), n_evals=len(history) if history else verdict.get('n_evals'),
                   epochs_run=d.get('epochs_run'), stopped_by=stopped, wall_time_sec=d.get('wall_time_sec',d.get('elapsed_sec')),
                   recipe_id=digest(recipe)[:16], test_windows=ntest, data_binding=data_binding,
                   dataset_recorded=d.get('dataset') or cfg.get('dataset'),
                   data_manifest_created=d.get('data_manifest_created'),
                   data_binding_kind='sha256' if re.fullmatch('[a-f0-9]{64}',str(data_binding)) else 'timestamp_or_unbound',
                   metric_convention=metric_convention, dataset_from=dataset_from,
                   diagnostic=verdict.get('status', ''), checkpoint_ref=checkpoint, checkpoint_sha256=cksha,
                   source_ref=source_refs[0], all_source_refs=source_refs,
                   artifact_sha256=digest(d),training_record_key=h,
                   source_count=len(source_refs), original_result_ids=old_ids[h], metrics_json=metrics,
                   config_json=cfg, source_release=selection.get('source') or d.get('source_run') or '',
                   numeric_verification='inventory_only; see linked audit records')
        rows.append(row)
        version.setdefault('run_count', 0)
        if vid not in versions:
            versions[vid] = version
            version['example_source'] = row['source_ref']
            version['run_names'] = set()
        versions[vid]['run_count'] += 1
        versions[vid]['run_names'].add(row['run_name'])
        if d.get('natural_grid_subset'):
            sub = dict(row)
            sub.update(record_id=h[:20]+'-natural', evidence_type='subset_reevaluation',
                       protocol='natural_grid_17224', test_windows=17224,
                       metrics_json=d['natural_grid_subset'])
            for k in ('cohen_kappa','pr_auc','average_precision','auroc','balanced_acc','weighted_f1','accuracy'):
                sub[k] = d['natural_grid_subset'].get(k)
            sub['primary_value'] = sub.get(primary)
            rows.append(sub)

    trials = []
    for path in table_sources:
        with path.open(newline='') as f:
            for index, d in enumerate(csv.DictReader(f, delimiter='\t'), 2):
                if not d:
                    continue
                trials.append(dict(record_id=digest([str(path),index,d])[:20], evidence_type='synthetic_or_readout_trial',
                                   study=path.relative_to(MAINT).parts[0],
                                   experiment=d.get('experiment') or d.get('variant') or d.get('run') or '',
                                   reported_status=d.get('status') or d.get('decision') or '',
                                   metric_name=d.get('metric_name') or ('curve_r2' if 'curve_r2' in d else 'reported_metric'),
                                   metric_value=d.get('metric', d.get('curve_r2')),
                                   curve_r2=d.get('curve_r2'), detection_auroc=d.get('detection_auroc'),
                                   source_ref=str(path)+f':{index}', artifact=d.get('artifact'), metrics_json=d))
    studies = []
    # Saved numeric probe results have no clinical dataset and remain separate.
    seen_scores = set()
    for s in sources:
        if Path(s['path']).name != 'scores.json' or '/smoke' in s['path']:
            continue
        h = digest(s['document'])
        if h in seen_scores:
            continue
        seen_scores.add(h)
        studies.append(dict(record_id=h[:20],record_kind='frozen_probe_result',
                            study=next((x for x in Path(s['path']).parts if x.startswith('legacy-triaxial')), 'frozen_probe'),
                            experiment=Path(s['path']).parent.name,metrics_json=s['document'],
                            source_ref=s['host']+':'+s['path']))
    for directory in sorted(MAINT.iterdir()):
        if not directory.is_dir() or directory.name == TASK.name:
            continue
        if not directory.name.startswith(('autoresearch','cfc-','pac-','original-cfft','legacy-',
                                           'own-model','explanation-audit','chbmit-','cbramod-',
                                           'lenovo-copula','no-training-resolution','comparison-correction',
                                           'pretrained145','iiic-','native-recovery','tuev-tokenizer')):
            continue
        reports = [p for p in [directory/'report.md',directory/'notes.md',directory/'task_plan.md'] if p.exists()]
        if not reports:
            continue
        report = reports[0]
        title = next((s.lstrip('# ').strip() for s in report.read_text(errors='replace').splitlines() if s.startswith('#')), directory.name)
        studies.append(dict(study=directory.name, title=title, record_kind='study_index',
                            trial_rows=sum(t['study']==directory.name for t in trials),
                            source_ref=str(report), other_notes=[str(p) for p in reports[1:]]))
    studies.extend(trials)
    checkpoints = {}
    archive_path = MAINT/'hf-checkpoint-archive-20261026/archive-manifest.json'
    archive_manifest = json.loads(archive_path.read_text())
    input_hashes[str(archive_path)] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    for item in archive_manifest['files']:
        key = (item['source_host'], item['path'])
        checkpoints[key] = dict(host=key[0], checkpoint_ref=key[1], bytes=item.get('size'),
                                sha256=item.get('sha256'), storage='HF archived snapshot',
                                object_ref=archive_manifest['repo_id']+'/'+item.get('object_path',''),
                                source_ref=str(archive_path), current_existence='not_rechecked')
    with (OLD/'checkpoints.csv').open(newline='') as f:
        for d in csv.DictReader(f):
            key=(d['host'],d['path'])
            if key not in checkpoints:
                checkpoints[key]=dict(host=key[0],checkpoint_ref=key[1],bytes=d['bytes'],
                                      sha256='',storage='historical_location',source_ref=str(OLD/'checkpoints.csv'),
                                      current_existence='not_rechecked')
    for s in sources:
        d=s['document']
        if s['host']=='local' and d.get('selected_checkpoint_sha256'):
            local_checkpoint=Path(s['path']).parent/'best.pt'
            if local_checkpoint.exists():
                key=('local',str(local_checkpoint))
                checkpoints[key]=dict(host='local',checkpoint_ref=key[1],sha256=d['selected_checkpoint_sha256'],
                                      bytes=local_checkpoint.stat().st_size,storage='verified_local_completed_run',
                                      source_ref=s['path'],current_existence='checked_local')
        if Path(s['path']).name=='selection.json' and d.get('checkpoint_sha256'):
            key=(s['host'],str(Path(s['path']).parent/'selected.pt'))
            checkpoints[key]=dict(host=key[0],checkpoint_ref=key[1],sha256=d['checkpoint_sha256'],
                                  storage='evaluation_snapshot',source_ref=s['host']+':'+s['path'],
                                  current_existence='selection_record')
        for item in d.get('current_runs', []) if isinstance(d,dict) else []:
            if isinstance(item,dict) and item.get('checkpoint') and item.get('sha256'):
                key=(s['host'],item['checkpoint'])
                checkpoints[key]=dict(host=key[0],checkpoint_ref=key[1],sha256=item['sha256'],
                                      bytes=item.get('size'),storage='checked_run_archive',
                                      source_ref=s['host']+':'+s['path'],current_existence='archive_receipt')
    conflicts=[]
    grouped=defaultdict(list)
    for row in rows:
        if row['evidence_type']=='clinical_run' and row['primary_value'] is not None:
            grouped[(row['dataset'],row['model_version'],row['recipe_id'],row['seed'],row['protocol'],str(row['data_binding']))].append(row)
    for key, group in grouped.items():
        if len({compact(r['metrics_json']) for r in group})>1:
            for row in group:
                conflicts.append(dict(conflict_id=digest(key)[:16],dataset=row['dataset'],
                                      model_display=row['model_display'],seed=row['seed'],
                                      protocol=row['protocol'],record_id=row['record_id'],
                                      primary_metric=row['primary_metric'],primary_value=row['primary_value'],
                                      status='needs_source_review; no winner selected',source_ref=row['source_ref']))
    # Preserve all original conflict groups, even if the new grouping is more specific.
    for item in json.loads((OLD/'seed-conflicts.json').read_text()):
        conflicts.append(dict(conflict_id='historical-'+digest(item)[:16],dataset=item.get('dataset'),
                              run_name=item.get('name'),seed=','.join(sorted(item.get('seed_result_ids',{}))),
                              status='historical_seed_conflict',
                              source_ref=str(OLD/'seed-conflicts.json'), details=item))
    rows.sort(key=lambda r:(r['dataset'],r['model_display'],str(r['seed']),r['run_name'],r['record_id']))
    for value in versions.values():value['run_names']=sorted(value['run_names'])
    write_csv('runs.csv',rows)
    write_csv('models.csv',sorted(versions.values(),key=lambda d:(d['model_display'],d['model_version'])))
    write_csv('studies.csv',studies)
    write_csv('checkpoints.csv',list(checkpoints.values()))
    write_csv('conflicts.csv',conflicts,('conflict_id','dataset','model_display','seed','status'))
    with (OUT/'source-records.jsonl').open('w') as f:
        for original_hash,document in sorted(original_documents.items()):
            h=result_key(document)
            f.write(json.dumps(dict(artifact_sha256=original_hash,training_record_key=h,
                                   document=document,sources=artifact_sources[h]),ensure_ascii=False)+'\n')
    summary=dict(updated_at=datetime.now(timezone.utc).astimezone().isoformat(),
                 historical_unique_results=len(wrappers),clinical_records=len(rows),
                 model_versions=len(versions),study_and_trial_records=len(studies),
                 trial_rows=len(trials),checkpoint_locations=len(checkpoints),conflict_rows=len(conflicts),
                 conflict_groups=len({x['conflict_id'] for x in conflicts}),
                 by_status=dict(Counter(r['status'] for r in rows)),by_dataset=dict(Counter(r['dataset'] for r in rows)),
                 input_files=len(input_hashes),input_hashes=input_hashes,parse_errors=errors,
                 remote_collection=(json.loads((TASK/'collected/latest-collection.json').read_text())
                                    if (TASK/'collected/latest-collection.json').exists()
                                    else '145 2026-11-05 16:09 Shanghai; Torch authentication unavailable'),
                 limits=['Historical and later metadata snapshots combined; not a new metric recomputation.',
                         'Historical checkpoint paths are recorded locations, not current availability claims.',
                         'Different protocols, native metric definitions and interim evaluations stay separate.',
                         'Synthetic/readout trials are in studies.csv, not pooled with clinical scores.'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    assert len({r['record_id'] for r in rows})==len(rows)
    assert set(w['result_id'] for w in wrappers)==set(x for r in rows for x in r['original_result_ids'])
    assert not errors,errors
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_hashes','by_dataset')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
