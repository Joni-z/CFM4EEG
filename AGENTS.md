# Repository working rules

- Use `Zhizhe` as both Git author and committer. Do not add co-author trailers.
- On a day with changes, review the diff, run the relevant checks, and commit
  completed work before ending the session. Do not create empty daily commits.
- Push completed, reviewed commits to `origin/main` before ending the session
  so GitHub reflects the current code and documentation.
- Keep `main` as the active development branch. Remove completed or superseded
  branches after checking their contents and preserving any unique history in
  an external Git bundle. Keep raw EEG, checkpoint files, runtimes and old
  workspaces under ignored `.local/`; publish result metadata and indexes in
  `experiments/`.
- Keep changes small and traceable to original implementations. Never rewrite
  a frozen execution release or replace recorded experiment results during
  repository maintenance.

## Canonical project location

- The active Mac mini project and Git checkout is `/Users/mr.z/CFM4EEG`.
  Do not create new CFM4EEG/PACLock sibling workspaces in the home directory.
- Reuse `.local/workspace/maintenance/` for local task evidence and
  `.local/legacy/` for the historical archive. Preserve recorded source paths
  in frozen documents; consult `.local/workspace/maintenance/project-consolidation-20261108/move-map.json`
  when locating moved local files.
- The shared experiment index is `experiments/runs.csv`; its workbook and
  GitHub pages are derived views. Update the existing ledger rather than
  starting a second result spreadsheet. Before comparing results, check
  protocol, initialization, recipe and metric definitions.
- After importing completed results, rebuild with the three scripts documented
  in `experiments/README.md`, verify record preservation, and commit/push the
  result metadata so the user can read it on GitHub. Never upload raw EEG,
  checkpoint binaries, environments or credentials to Git.
