# Original code and baseline entrypoints

Exact repositories, commits and author entry files are listed in [sources.json](../upstream/sources.json). `python scripts/fetch_upstream.py biot` restores that author checkout under `vendor/biot`; substitute the desired model name. The fetcher refuses to overwrite an existing checkout at another revision. Model weights are separate inputs, not silently downloaded by the trainer.

## Entry map

| Baseline | Local entry under `scripts/rebuild/` | Actual original code used | Declared adaptation |
|---|---|---|---|
| BIOT / FFCL | `native_biot_entry.py` | Author `supervised`, LightningModule, optimizer, Trainer and checkpoint selection | Shared arrays exposed to native loaders; seed0; CHB selects the existing native CHB preparation branch |
| CBraMod, TUEV | `native_cbramod_entry.py` | Author `finetune_main.main`, Trainer and Evaluator | Shared arrays, input units and scoped boolean parsing |
| CBraMod, CHB/TUAB | `native_cbramod_clinical_entry.py` | Same original main/Trainer/Evaluator, original task heads | CHB inverse-Q95 sidecar; TUAB stored µV/100; original validation selection |
| EEGConformer | `native_eegconformer_clinical_entry.py` | Author model forwards and extracted Adam/CE update block | Clinical tasks are new: explicit shape changes, no MI S&R, validation selection and clinical budget |
| EEGPT, CHB | `run_eegpt_chb_official.py` | Original TUAB main/model/engine, trainable encoder with explicit scratch or pretrained initialization | New binary task; input convolution16→20 instead of23→20; full sequential validation/test |
| EEGPT, TUEV | `run_eegpt_tuev_official.py` | Original23-channel model, main, pickle reader and engine | Original native AMP branch instead of the released shell’s DeepSpeed backend |
| LaBraM | `run_labram_original.py` | Original main/engine, AdamW, cosine schedule and native checkpoint functions | CHB/ADFD scratch uses ordinal slots; pretrained CHB uses explicit fresh bipolar positions; TUEV uses native23 inputs and disclosed accumulation |
| REVE | `run_reve_native.py` | Original downstream LP/FT loops, StableAdamW, warmup and criterion | Shared-cohort array/coordinate bridge; scratch runs full FT from random initialization |
| TFM-Tokenizer | `run_tfm_chain.py`, `run_tfm_stage.py` | Original tokenizer, MTP and downstream Lightning modules | Target-only train data for single-dataset stages; explicit stage dependencies |

The FFCL implementation here is the released BIOT benchmark implementation. It is not presented as a separately certified replication of FFCL’s own original experiment.

The translated legacy baseline architectures and generic training dispatch have been removed. The two CBraMod adapters correspond to different original task heads and input contracts; neither reimplements its optimizer or training loop.

## Explicit commands

All commands run from the repository root, in the model’s execution environment. Output directories must be fresh. Config paths and environment variables must refer to accepted data; the smoke must not read held-out test payloads.

Conformer, one GPU:

```bash
python scripts/rebuild/native_eegconformer_clinical_entry.py \
  --vendor vendor/eeg-conformer --dataset tuev \
  --data "$PACLOCK_PROC/processed/tuev" --out "$CFM_REBUILD_ROOT/smoke/conformer-tuev" --smoke-steps 4
python scripts/rebuild/native_eegconformer_clinical_entry.py \
  --vendor vendor/eeg-conformer --dataset tuev \
  --data "$PACLOCK_PROC/processed/tuev" --out "$CFM_REBUILD_ROOT/runs/conformer-tuev" \
  --smoke-proof "$CFM_REBUILD_ROOT/smoke/conformer-tuev/SMOKE_PASSED.json"
```

Use `chbmit` or `tuab` with the corresponding data directory. The config records the fixed seed0, author Adam/CE and clinical budget; the CLI implements those values explicitly.

LaBraM, one GPU with the configuration’s documented accumulation:

```bash
python scripts/rebuild/run_labram_original.py configs/baselines/labram/chbmit-scratch.json \
  --vendor vendor/labram --out "$CFM_REBUILD_ROOT/smoke/labram-chbmit" --smoke
python scripts/rebuild/run_labram_original.py configs/baselines/labram/chbmit-scratch.json \
  --vendor vendor/labram --out "$CFM_REBUILD_ROOT/runs/labram-chbmit" \
  --smoke-proof "$CFM_REBUILD_ROOT/smoke/labram-chbmit/smoke_pass.json"
```

For native23 LaBraM, set `PACLOCK_NATIVE` to the parent of the fully verified `tuev-labram23/` cache. CHB and ADFD ordinal slots carry no anatomical-electrode claim.

Native23 EEGPT uses the same `PACLOCK_NATIVE` cache root, two GPUs, per-rank batch400 and30epochs:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/rebuild/run_eegpt_tuev_official.py configs/baselines/eegpt/tuev-native23-scratch.yaml \
  --out "$CFM_REBUILD_ROOT/smoke/eegpt-tuev23" --seed 0 --smoke-steps 4
```

The native23 reader verifies the original preprocessor/consumer hashes and accepted transfer. Formal execution uses the same command with a fresh `runs/` output and without `--smoke-steps`; the job must depend on that smoke succeeding. CHB’s two-rank entry uses the common launcher. Never replace a trainable scratch encoder with a frozen random encoder.

The generic `slurm/rebuild/torch_native.slurm` defaults to one GPU. For CHB EEGPT, submit it with `sbatch --gres=gpu:2 ...`; the launcher refuses to start its two ranks with fewer than two visible devices.

CHB EEGPT also accepts an explicit `--resume` with a trusted local full training checkpoint. Continuation restores the optimizer and scaler, starts at the saved epoch plus one, and preserves the best validation-selected weights across segments. Keep the existing output directory and its validation history. The adapter corrects the upstream hardcoded resume epoch; model and training-engine sources remain unchanged. Upstream checkpoints do not store RNG state, so continuation does not reproduce the interrupted dropout trajectory exactly.

The default arithmetic is FP16. An explicit `--precision bf16` or `--precision fp32` applies consistently to training, validation and final testing and is recorded as a runtime adaptation. Verify a fresh recovery smoke with the same checkpoint and precision before continuation; a successful forward diagnostic alone does not establish that a training instability has been resolved.

TFM single-dataset work consists of tokenizer100epochs, masked-token prediction50epochs, then downstream50epochs. Run `run_tfm_chain.py CONFIG --plan` for the stage paths; each stage uses `--stage tokenizer|mtp|finetune --seed 0 --mode auto --resume auto`. The first two require2GPUs; downstream requires1. A stage’s `completed.json`, not a partial checkpoint, permits the next stage. Released multiple-dataset weights have separate exposure/protocol checks.

## Runtime differences

The package lists under `environments/` document actual executed environments. They are not a claim that the authors supplied identical locks. BIOT/FFCL use isolated Python3.9, Torch1.13.1 and Lightning1.6.4; on Ada, the verified per-process CUDA11.8 cuFFT replacement is retained in the execution receipt. Other baselines use their recorded modern runtime and explicit compatibility branches. Do not merge these into one silently upgraded environment.

## Released-pretrained finetuning

See [pretrained baseline contracts](PRETRAINED-BASELINES.md) for the TUEV/CHB seed-0 configurations, strict weight loading, original task recipes, runtime adaptations and cohort distinctions. These entries load released weights; they do not run pretraining.

## IIIC scratch tasks

[IIIC task contracts](IIIC-BASELINES.md) document the four native new-task adaptations, their full ten-second input windows, exact accepted cohort, fixed recipes and smoke gates. They reuse original author code and explicitly adapt new input/task interfaces.
