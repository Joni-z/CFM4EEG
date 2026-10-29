# Implementation compatibility

The project/model name is CfFT. Python names `paclock_bench`, `paclock`,
`triaxial.py` and existing checkpoint tensor keys are retained for compatibility.
The selected configuration determines the actual frontend and encoder axes.

Use the [model guide](../../../docs/MODEL.md) for tokenizer variants and source
locations, the [CFC guide](../../../docs/CFC_OPERATOR.md) for measurement, and the
[audit guide](../../../docs/AUDIT.md) for immutable validation records.

Historical configurations and checkpoints retain their recorded source releases.
`measured_pac` uses the earlier first-harmonic estimator. The active `cfc` frontend
uses the unchanged dual-harmonic operator with original CfFT rotation and local
fallback. Discontinued integration options remain in the source tags linked from
the model guide.
