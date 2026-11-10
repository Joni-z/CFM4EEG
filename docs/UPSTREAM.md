# Baseline reproduction

## Sources and entrypoints

[upstream/sources.json](../upstream/sources.json) records author repositories,
exact commits and original entry files. Restore a pinned checkout with
`python scripts/fetch_upstream.py biot`, substituting the source key as needed.
The fetcher refuses to overwrite a different revision. Released weights are
separate inputs; training does not silently download them.

| Baseline | Entry under `scripts/rebuild/` | Author code and explicit adaptation |
|---|---|---|
| BIOT / FFCL | `native_biot_entry.py` | Original supervised model, optimizer and Lightning trainer; shared arrays exposed through native loaders, seed 0 and native CHB preparation branch |
| CBraMod, TUEV / IIIC | `native_cbramod_entry.py` | Original main, trainer and evaluator; shared arrays, scoped boolean parsing and the IIIC head-width adaptation below |
| CBraMod, CHB / TUAB | `native_cbramod_clinical_entry.py` | Original binary heads and trainer; CHB inverse-Q95 unit restoration, TUAB stored µV/100 |
| EEGConformer | `native_eegconformer_clinical_entry.py` | Author model and Adam/CE update block; new clinical tasks require explicit shape/budget changes and omit MI-specific S&R augmentation |
| EEGPT, CHB / IIIC | `run_eegpt_chb_official.py` | Original TUAB main/model/engine; 16→20 channel convolution, task output head and sequential validation/test |
| EEGPT, TUEV | `run_eegpt_tuev_official.py` | Original TUEV main/model/engine; shared16 changes the input convolution width, native23 retains its own cohort; native AMP replaces the released shell's DeepSpeed backend |
| LaBraM | `run_labram_original.py` | Original main/engine, AdamW and schedule; explicit spatial-slot initialization, units and single-GPU accumulation |
| REVE | `run_reve_native.py` | Original downstream LP/FT loops, StableAdamW, warmup and criterion; shared arrays/coordinates; scratch uses full FT from random initialization |
| TFM-Tokenizer | `run_tfm_chain.py`, `run_tfm_stage.py` | Original tokenizer, MTP and downstream modules; single-dataset stages use only target training data and explicit dependencies |

FFCL is the implementation released in the BIOT benchmark, not a separately
certified replication of FFCL's own experiment. The two CBraMod adapters retain
different native task heads. No baseline falls back to the project trainer.

[DATA.md](DATA.md) defines cohorts and unit conversions. Shared16 TUEV uses
68,445 / 15,487 / 29,421 training/validation/test windows. Native23 TUEV is a
separate protocol and cannot populate the same-test comparison. Shared-cohort
training with author code is an adaptation, not an identical paper protocol.

## Execution

Run from the repository root in the model's [recorded environment](../environments/README.md),
with allocated GPUs and external data/output paths:

```bash
export PACLOCK_PROC=/path/to/data
export CFM_REBUILD_ROOT=/path/to/outputs
```

The configurations and commands below use seed 0. Use fresh output directories
except for a documented resume. The common launcher
checks the exact configuration hash in [accepted_configs.json](../review/accepted_configs.json)
and requires a matching smoke receipt. A file under `configs/baselines/` is not
itself approval. Direct LaBraM and TUEV EEGPT CLIs enforce their own source,
input and smoke checks; the common registry is not a universal gate.

### Common launcher

BIOT, FFCL and CBraMod use this interface; for example:

```bash
python scripts/rebuild/launch.py configs/baselines/biot/tuev-scratch.yaml 0 smoke
python scripts/rebuild/launch.py configs/baselines/biot/tuev-scratch.yaml 0 train
```

The generic [Torch job](../slurm/rebuild/torch_native.slurm) defaults to one GPU.
When submitting smoke and training separately, add `--dependency=afterok:SMOKE_JOB_ID`
to the formal submission; the job file does not add dependencies automatically.

### EEGConformer

One GPU, explicit clinical interface:

```bash
python scripts/rebuild/native_eegconformer_clinical_entry.py \
  --vendor vendor/eeg-conformer --dataset tuev \
  --data "$PACLOCK_PROC/processed/tuev" --out "$CFM_REBUILD_ROOT/smoke/conformer-tuev" --smoke-steps 4
python scripts/rebuild/native_eegconformer_clinical_entry.py \
  --vendor vendor/eeg-conformer --dataset tuev \
  --data "$PACLOCK_PROC/processed/tuev" --out "$CFM_REBUILD_ROOT/runs/conformer-tuev" \
  --smoke-proof "$CFM_REBUILD_ROOT/smoke/conformer-tuev/SMOKE_PASSED.json"
```

Use `chbmit` or `tuab` with its matching data directory. The CLI implements the
fixed seed-0 clinical adaptation; it is not the author's published MI recipe.

### LaBraM

One GPU with configuration-specific accumulation:

```bash
python scripts/rebuild/run_labram_original.py configs/baselines/labram/chbmit-scratch.json \
  --vendor vendor/labram --out "$CFM_REBUILD_ROOT/smoke/labram-chbmit" --smoke
python scripts/rebuild/run_labram_original.py configs/baselines/labram/chbmit-scratch.json \
  --vendor vendor/labram --out "$CFM_REBUILD_ROOT/runs/labram-chbmit" \
  --smoke-proof "$CFM_REBUILD_ROOT/smoke/labram-chbmit/smoke_pass.json"
```

Released-weight runs use the corresponding pretrained JSON and the same CLI.
For native23, `PACLOCK_NATIVE` points to the parent of the verified
`tuev-labram23/` cache. Scratch CHB/ADFD spatial slots carry no anatomical claim.

### EEGPT

Both TUEV protocols require the explicit CLI. Shared16 released-weight example,
two GPUs, batch 400 per rank and 30 epochs:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/rebuild/run_eegpt_tuev_official.py configs/baselines/eegpt/tuev-shared16-pretrained.yaml \
  --out "$CFM_REBUILD_ROOT/smoke/eegpt-tuev" --seed 0 --precision bf16 --smoke-steps 4
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/rebuild/run_eegpt_tuev_official.py configs/baselines/eegpt/tuev-shared16-pretrained.yaml \
  --out "$CFM_REBUILD_ROOT/runs/eegpt-tuev" --seed 0 --precision bf16 \
  --smoke-proof "$CFM_REBUILD_ROOT/smoke/eegpt-tuev/smoke.json"
```

CHB released-weight runs use `run_eegpt_chb_official.py` and
`configs/baselines/eegpt/chbmit-pretrained.yaml` with the same two-rank invocation,
explicit BF16 and smoke proof; their native budget is batch 100 per rank and
50 epochs. The common launcher does not forward the proof required by these
pretrained formal runs. CHB scratch can use that launcher with two allocated GPUs;
request `sbatch --gres=gpu:2` when using the generic Torch job.

Native23 scratch uses `tuev-native23-scratch.yaml`, the verified `PACLOCK_NATIVE`
cache and the runner's default FP16, or an explicitly recorded precision
adaptation. Its legacy reader constructs all native
splits during smoke, including test payloads; that smoke is not evidence of
held-out input isolation. Shared16 smoke constructs training/validation only.
Native23 remains a separately labeled protocol.

CHB continuation accepts `--resume` with a trusted full training checkpoint in
the same output directory. It restores optimizer/scaler, starts after the saved
epoch and preserves validation-best weights/history. Upstream checkpoints lack
RNG state, so dropout trajectories are not exactly reproducible. Use one explicit
precision (`fp16`, `bf16` or `fp32`) consistently in training, validation and test,
and obtain a matching recovery smoke before continuation. Scratch encoders must
remain trainable.

### REVE

Use the original single-GPU stages. A released-weight smoke may need more than
four steps to demonstrate updates during AMP warmup; the explicit interface
allows this without changing the formal recipe:

```bash
python scripts/rebuild/preflight.py --config configs/baselines/reve/tuev-pretrained.yaml --mode smoke
python scripts/rebuild/run_reve_native.py configs/baselines/reve/tuev-pretrained.yaml \
  --seed 0 --out "$CFM_REBUILD_ROOT/smoke/tuev-reve_pretrained_rebuild20261021/seed0" --smoke-steps 16
python scripts/rebuild/preflight.py --config configs/baselines/reve/tuev-pretrained.yaml --mode train
python scripts/rebuild/run_reve_native.py configs/baselines/reve/tuev-pretrained.yaml \
  --seed 0 --out "$CFM_REBUILD_ROOT/runs/tuev-reve_pretrained_rebuild20261021/seed0"
```

The smoke checks actual optimizer steps, classifier/FT-backbone updates and
checkpoint reload; query movement is recorded separately during short warmup.
Recovery snapshots retain model/optimizer state, but this adapter does not resume
native loop counters. An interrupted run is not a completed result.

### TFM-Tokenizer

Single-dataset training has tokenizer 100 epochs → MTP 50 → downstream 50.
Each command handles one stage, with its own smoke and completion gate:

```bash
python scripts/rebuild/run_tfm_chain.py configs/deferred/tfm/tuev-single.yaml --plan
python scripts/rebuild/run_tfm_chain.py configs/deferred/tfm/tuev-single.yaml \
  --stage tokenizer --seed 0 --mode auto --resume auto
```

After completion, run the same stage command with `--stage mtp`, then
`--stage finetune`. Allocate two GPUs for tokenizer/MTP and one for finetuning.
Only a matching `completed.json` with complete steps and verified artifacts
permits the next stage. Exit 75 means resume the same incomplete stage; a partial
checkpoint does not authorize its dependent. Released multiple-dataset weights
have separate exposure/protocol checks and run finetuning only.

## Released-pretrained initialization

These configurations load released weights and train downstream; they do not
run multi-dataset pretraining. Keep scratch and pretrained results separate.

| Model | Configs under `configs/baselines/<model>/` | Weight input and native recipe |
|---|---|---|
| BIOT | `tuev-pretrained.yaml`, `chbmit-pretrained.yaml` | `$PACLOCK_BIOT_WEIGHTS`; SHHS+PREST encoder, capacity 18 for 16 actual channels; native Adam, early stopping and classifier |
| CBraMod | `tuev-pretrained.yaml`, `chbmit-pretrained.yaml` | `$PACLOCK_CBRAMOD_WEIGHTS`; strict backbone load, pretraining projection removed; original head and 50-epoch full finetuning |
| EEGPT | `tuev-shared16-pretrained.yaml`, `chbmit-pretrained.yaml` | `$PACLOCK_EEGPT_PRETRAINED`; complete target encoder checked tensor by tensor; original two-rank recipes with explicit BF16 |
| LaBraM | `tuev-shared16-pretrained.json`, `chbmit-pretrained.json` | `$CFM_VENDOR/labram/checkpoints/labram-base.pth`; student backbone, native engine and validation-accuracy selection; batch 64 × 8 accumulation |
| REVE | `tuev-pretrained.yaml`, `chbmit-pretrained.yaml` | Local `vendor/reve_weights/`; SHA-bound safetensors encoder/query; original LP then FT, StableAdamW, warmup and validation-balanced-accuracy selection |

Supply weight paths before smoke. `CFM_VENDOR` expands LaBraM's weight path;
ordinary `vendored()` lookup still uses this repository's `vendor/` directory.
The config fixes each checkpoint SHA256. Missing active tensors or unequal loaded
values fail before training; task classifiers retain random initialization.

CHB is a new task for EEGPT and LaBraM. EEGPT changes the TUAB input convolution
from 23 to 16 inputs while keeping its twenty latent positions and native
finetuning architecture. Its pretrained configs use BF16 because an FP16 pilot
skipped optimizer updates with nonfinite gradients. Model, optimizer, schedule
and batch remain native; the arithmetic adaptation is explicit.

REVE's TUH convention is physical µV/100 at 200 Hz, equivalent to raw volts
multiplied by 10,000. CHB restores physical units from its inverse-Q95 sidecar
before that scaling. Encoder tensors and the released query are checked exactly.
The original LP stage passes its last weights to FT; final testing restores the
FT validation-best checkpoint. Config `stage.dropout` is unused upstream and the
effective classifier dropout is 0; effective clipping is 2.

### LaBraM loading details

The released base checkpoint SHA256 is
`7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.
A tensor-only serialization bridge preserves the released model values while
avoiding legacy metadata incompatibility. All shared tensors are checked after
the original loader. The classifier/pooling LayerNorm stay task-initialized;
the native loader discards pretraining-only masked-token head/norm.
Final selected-checkpoint evaluation saves labels, logits and probabilities in
`test_scores.npz`; smokes save validation predictions separately.

Shared16 TUEV and CHB use fresh bipolar spatial slots 1–16, retaining pretrained
CLS/time embeddings and the shared backbone. Bipolar inputs are not relabeled as
monopolar electrodes. The author's CHB lookup maps names to positions 121–136,
which exceed the released table's valid range 0–128; it cannot be applied
unchanged. Physical µV is restored before the native engine divides by 100.

The adapted command retains 50 epochs, AdamW LR `5e-4`, weight decay 0.05,
warmup 5, layer decay 0.65 and drop path 0.1. Accumulating eight microbatches of
64 preserves global batch 512, without claiming distributed numerical identity.
See the pinned [README](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/README.md)
and [channel lookup](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/utils.py).

## IIIC task adaptations

The accepted cohort has 73,637 / 15,778 / 15,780 training/validation/test windows,
16 bipolar channels, labels 0–5 and complete 10-second / 2,000-sample inputs
stored as µV/100. These are new downstream task adaptations, not reproductions
of published IIIC commands. No cropping, temporal compression or replacement
test cohort is part of the interface.

| Baseline | Input and model interface | Configured recipe |
|---|---|---|
| BIOT | Restore physical µV; original `iiic_load`, native Q95 and one-hot labels; remove only its returned dataset-id flag | Scratch capacity 16; batch 128, Adam `1e-3`, weight decay `1e-5`, 100 epochs, patience 5, 32 workers |
| CBraMod | Stored µV/100 reshaped into ten patches; first classifier input 16,000→32,000; native hidden widths 1,000/200 and six outputs | Batch 64, AdamW `1e-4`, weight decay 0.05, 50 epochs, multi-LR, clip 1, smoothing 0.1, native cosine |
| LaBraM | Restore physical µV before native /100; fresh scratch spatial slots without anatomical claims | TUAB budget: 50 epochs, batch 64 × 8 accumulation, AdamW `5e-4`, weight decay 0.05, warmup 5, scratch layer decay 1 |
| EEGPT | Original ten-second TUAB decoder with 16→20 convolution, six outputs and a trainable random encoder; restore µV before native /100 | Two ranks, batch 100/rank, 50 epochs, AdamW `5e-4`, weight decay 0.05, warmup 5, layer decay 0.65, explicit BF16 |
| REVE | Original full-FT stage, random initialization, stored shared arrays and explicit position convention | Batch 256, 50 epochs, patience 10, StableAdamW `1e-4`, weight decay 0.01, warmup 5, mixup; effective clip 2 and classifier dropout 0 |

Configs are `configs/baselines/<model>/iiic-scratch.*`. Use each model's interface
above, preserving the IIIC config and precision. LaBraM binds its native
`smoke_pass.json`; common entries bind source/config/data in `smoke.json`.
[Audit rules](AUDIT.md) define test isolation, checkpoint selection, BIOT terminal
versus validation-best outputs and comparable metric definitions.
