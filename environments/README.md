# Execution environments

These package-name/version inventories were read from the isolated145 environments used by the native entrypoints. They document observed runtimes; they are not original-author lockfiles or a claim of portability across CUDA drivers. TFM is import-checked; no completed TFM training is implied.

BIOT/FFCL use Python3.9, Torch1.13.1, Lightning1.6.4. Ada execution additionally uses the validated CUDA11.8 cuFFT library via a per-process preload; that external library and its SHA are in the original run receipt, not this pip inventory. LaBraM and EEGPT use the official native AMP branches with their recorded modern runtime adaptations.

Keep model environments separate. Restoring an environment does not bypass source/data checks or GPU smoke. CPU development checks can use `requirements-dev.txt` at the repository root.
