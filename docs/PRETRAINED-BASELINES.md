# Released-pretrained baseline finetuning

These configurations load the authors' released pretraining checkpoints and run their downstream training code. They do not launch pretraining. Every cell uses seed 0, a fresh output directory and a real GPU smoke before formal execution. Scratch results remain separate.

| Model | Configurations under `configs/baselines/<model>/` | Initialization and native recipe |
|---|---|---|
| BIOT | `tuev-pretrained.yaml`, `chbmit-pretrained.yaml` | Released SHHS+PREST encoder; channel-token capacity 18 with 16 actual inputs, as in the author's example; native Adam, early stopping and downstream classifier |
| CBraMod | `tuev-pretrained.yaml`, `chbmit-pretrained.yaml` | Strict native backbone loading followed by removal of the pretraining projection; native full finetuning, task head and 50-epoch recipe |
| EEGPT | `tuev-shared16-pretrained.yaml`, `chbmit-pretrained.yaml` | Complete released target encoder verified tensor by tensor; original two-rank TUEG full-finetuning recipes, 30 epochs for TUEV and 50 for the CHB adaptation |
| LaBraM | `tuev-shared16-pretrained.json`, `chbmit-pretrained.json` | Released student backbone; original finetuning engine and validation-accuracy selection; 64-sample microbatches accumulated eight times to preserve the published global batch of 512 |

## Data and task adaptations

The main-table TUEV configurations all use the same accepted shared 16-channel cohort (68,445/15,487/29,421 train/validation/test windows). EEGPT changes only the learned channel-convolution input width to 16 while preserving its native twenty latent positions; LaBraM uses fresh bipolar spatial slots. The retained `tuev-native23-pretrained` configurations are a separate original 23-channel cohort and must not be mixed into the shared16 comparison. All CHB cells use the corrected accepted shared data and explicit inverse-Q95 unit restoration where required by the native loader.

CHB is an added task for EEGPT and LaBraM. EEGPT retains the author's TUAB decoder and changes its learned input-channel convolution from 23 to 16 inputs. LaBraM uses fresh task-specific bipolar spatial slots because the original bipolar name lookup exceeds the released position table; shared backbone and CLS/time embeddings remain pretrained. See [LaBraM's loading contract](LABRAM-PRETRAINED.md).

## Loading and numerical checks

Checkpoint SHA256 is fixed in each configuration. Missing weights or unexpected active tensor omissions fail before training. Audits compare loaded values to released values, then verify real optimizer updates and exact checkpoint reload. New task classifiers remain randomly initialized.

EEGPT's four-step FP16 pilot had nonfinite gradients and skipped every optimizer update, so the pretrained configurations explicitly use BF16. Both BF16 two-rank smokes passed with finite gradients and updated encoder/classifier weights. This runtime adaptation is recorded; the model, optimizer, schedule and batch are unchanged. BIOT retains its isolated legacy environment and documented cuFFT compatibility wrapper. CBraMod uses its original full-precision training; LaBraM retains native AMP.

## Selection and artifacts

The author's native loop and selection rule are retained. BIOT's released code defaults to a terminal checkpoint despite its paper's validation-best description; native terminal outputs and independently archived validation-best checkpoints remain distinct. A validation-selected reevaluation is stored separately and must use only validation metrics to choose the checkpoint.

Native results, validation histories, source/config/data/weight hashes, selected checkpoints and prediction artifacts remain outside the source repository. Passed execution snapshots are immutable. A new source or configuration requires a new smoke. On 145, the two EEGPT cells require two GPUs each and are queued without changing their per-rank batch.
