# Execution environments

These package/version inventories describe observed execution runtimes. They are
not author-provided lockfiles, portability guarantees or proof that a model
trained successfully. Source, data and GPU smoke receipts remain separate checks.

| Inventory | Runtime |
|---|---|
| [biot.txt](biot.txt) | BIOT / FFCL legacy dependencies |
| [conformer.txt](conformer.txt) | EEGConformer clinical interface |
| [eegpt.txt](eegpt.txt) | EEGPT native AMP compatibility runtime |
| [labram.txt](labram.txt), [labram-torch.txt](labram-torch.txt) | Recorded LaBraM executions on different hosts |
| [reve.txt](reve.txt) | REVE native downstream runtime |
| [tfm.txt](tfm.txt) | TFM tokenizer, MTP and downstream dependencies |

BIOT/FFCL retain Python 3.9, Torch 1.13.1 and Lightning 1.6.4. Ada execution also
uses a verified CUDA 11.8 cuFFT replacement through per-process preload; its
library and SHA are recorded in the execution receipt, not the pip inventory.
Other models retain their recorded native AMP or full-precision branches and
explicit runtime adaptations; do not silently combine them into one environment.

CPU development checks use [requirements-dev.txt](../requirements-dev.txt).
For commands and model-specific precision, see the [baseline guide](../docs/UPSTREAM.md).
