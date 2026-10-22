# Auditing an experiment

Read the experiment in this order:

1. `upstream/sources.json`: author repository, exact commit and original main/engine files.
2. `configs/baselines/<model>/`: dataset, scratch/pretrained setting, dimensions, optimizer, budget and declared adaptations. `configs/planned/` is not executable evidence.
3. `scripts/rebuild/`: the small local input/CLI bridge. Follow its import into the original `vendor/` main or engine; no generic baseline fallback exists.
4. The run’s dataset manifest, acceptance receipt and any build/transfer receipt: subject cohort, units and exact arrays/pickles.
5. The model-specific smoke receipt: actual optimizer updates, checkpoint reload, source/config/data binding and no test evaluation.
6. The run’s recorded CLI/environment, validation selection, stopping reason, final metrics and retained checkpoint. Use the author’s selection rule, including any documented difference between released code and paper text.

The generic project trainer and model factory construct CfFT only. Missing baseline code is an error, not permission to substitute another implementation. A first-evaluation peak is diagnostic information; it does not automatically invalidate a checkpoint selected under the original recipe.

## Repository and execution snapshots

This source tree contains current implementations and a compact config/source binding under `review/`. A completed run still belongs to its original immutable release, not to whatever code happens to be on the current branch.

The cleanup retained all old files and experiment records at `../history/repository-cleanup-20261022/`. The independently SHA-verified full snapshot and path mapping are under `../maintenance/repository-cleanup-20261022/`. Existing145/Torch releases, data, checkpoints and manuscripts were not rewritten. These are external records, not import paths or a second active implementation.

When creating a new release, copy only this source tree, restore the pinned upstream repositories, set external data/output roots, and obtain a new smoke for the exact source/config/data before formal training. An earlier release’s smoke must not be relabelled to bless changed files.

For local checks, use `python -m pytest -q` and `python -m tests.check_preprocessing`. Data-volume and GPU checks run under the corresponding compute allocation. Historical configs, run collection, plotting, spreadsheet updates and scheduler incidents are kept with the external records rather than mixed into training code.
