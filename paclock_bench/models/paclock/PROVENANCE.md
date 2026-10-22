# CfFT implementation

This package contains the project’s learned filterbank, analytic-signal features,
relation-conditioned tokenizer and Transformer encoder. The `paclock` module and
class names are retained for checkpoint compatibility; the project/model name is
CfFT.

Current configurations are under `configs/model/`. They select16bands,
quadrature envelope/waveform content, measured relations, temporal attention and
joint electrode–band row attention. Filename `triaxial.py` also contains the
older axis options; the selected configuration determines the actual encoder.

The repository cleanup did not change these tokenizer/encoder computations or
checkpoint tensor keys. Historical `_prev` and `_reference` copies are preserved
outside the active source tree. The pre-cleanup source snapshot and a fixed-input
forward/state comparison are recorded in the external maintenance directory.

Baseline reproduction uses original-author code and separate adapters; it does
not instantiate baselines through this model package. Dataset contracts are in
`docs/DATA.md`.
