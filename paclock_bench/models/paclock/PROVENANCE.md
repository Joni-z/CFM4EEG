# Implementation compatibility

The project/model name is CfFT. Python names `paclock_bench`, `paclock`,
`triaxial.py` and existing checkpoint tensor keys are retained for compatibility.
The selected configuration determines the actual frontend and encoder axes.

Use the [model guide](../../../docs/MODEL.md) for tokenizer variants and source
locations, the [CFC guide](../../../docs/CFC_OPERATOR.md) for measurement, and the
[audit guide](../../../docs/AUDIT.md) for immutable validation records.

Historical configurations and checkpoints retain their original identity.
`measured_pac` uses the earlier first-harmonic estimator; `cfc` uses the independent
dual-harmonic operator. Residual integration is opt-in through `cfc_fusion`;
rotation remains the default for earlier CFC configurations.
