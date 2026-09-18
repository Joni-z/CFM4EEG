# CFM4EEG / CroFreMo

**Project status: paused for archival closeout on 2026-09-19.** The current manuscript is frozen; no new training campaign or conference submission is planned until the project is explicitly resumed.

CroFreMo (Cross-Frequency Modulation) is a PAC-inspired EEG tokenizer. It learns frequency bands, computes local phase–amplitude moments, and uses measured relation directions to rotate paired envelope and waveform features. The last reference implementation uses 16 bands, quadrature content, and a two-axis encoder with temporal attention and joint electrode–band row attention. Legacy names such as `PACLock`, `paclock_bench`, `A128`, and `quad` remain in historical configurations and paths.

The frozen manuscript is titled **Cross-Frequency Modulation for EEG Tokenization**. The paper's Overleaf repository and this code repository are separate Git histories. The closeout archive provides one entry point linking both, their editable figures, experimental records and available weights.

## Start here

- [Closeout and restart guide](docs/CLOSEOUT-2026-09-19.md): what is preserved, where to look, and how to resume.
- [Documentation index](docs/INDEX.md): current guidance versus historical experiments.
- [Data protocols](docs/PROTOCOLS.md): corpus and loader definitions.
- [Previous README](docs/history/README-before-closeout.md): historical setup and invocation details.

## Source layout

| Path | Purpose |
|---|---|
| `paclock_bench/` | Models, loaders, metrics, training and runtime support |
| `paclock_bench/models/paclock/frontend/triaxial.py` | Learned filtering, analytic features, cross-frequency token construction; filename retained for compatibility |
| `configs/` | Dataset-, architecture-, host- and control-specific configurations |
| `preprocessing/` | Dataset preparation and manifests |
| `slurm/`, `scripts/`, `smoke/` | Resource-specific execution and checks |
| `docs/` | Protocols and dated research records |
| `runs/`, `results/` | Historical artifacts; availability varies by checkout |

The last two-axis runtime includes uncommitted changes beyond the default branch. Its live source/config snapshot and the Sean-control runtime snapshot are preserved in the private closeout archive. **Do not assume that cloning `main` alone recreates the frozen manuscript's full runtime.** Git bundles, working-tree patches and source snapshots are retained together.

## Experimental source of truth

Use original run artifacts, not manuscript cells or hand-maintained spreadsheets. The closeout ledger includes per-seed test metrics, validation selection, stopping reason, actual completed seed counts and artifact locations. Identical copies are deduplicated. Different artifacts for the same configuration/seed remain unresolved rather than selecting the higher score. Single-seed results have no inferred standard deviation.

A frozen manuscript is a versioned writing artifact, not an experimental database. Source-derived records and manuscript figure inputs are stored separately. Dataset/loader differences and compute truncation must be checked before comparing rows.

## Restarting

1. Read the closeout guide and recover the matching code/configuration, checkpoint and data manifest.
2. Inspect any runs that finished after the archive snapshot; the closeout collector only records results and never submits training.
3. Verify dataset paths and subject splits, then smoke-test the exact configuration on a compute node.
4. Re-establish the model and comparison protocol before launching new experiments.

Do not compute on cluster login nodes. On AMD, use a suitable single-GPU partition or fill all GPUs in a packed node allocation. B2 is a shared account and must be scoped by project directory. The 145 machine is shared: check device occupancy before use. All existing checkpoints are retained; no closeout task deletes model weights or dataset arrays.

No validated CroFreMo pretraining result or universal host-transfer improvement is claimed by this README. Supervised results, architecture controls, host replacements and exploratory pretraining are distinct records.
