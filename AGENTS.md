# Repository working rules

- Use `Zhizhe` as both Git author and committer. Do not add co-author trailers.
- On a day with changes, review the diff, run the relevant checks, and commit
  completed work before ending the session. Do not create empty daily commits.
- Push completed, reviewed commits to `origin/main` before ending the session
  so GitHub reflects the current code and documentation.
- Keep `main` as the active development branch. Remove completed or superseded
  branches after checking their contents and preserving any unique history in
  an external Git bundle. Keep experiment data and checkpoint files outside
  the active source tree.
- Keep changes small and traceable to original implementations. Never rewrite
  a frozen execution release or replace recorded experiment results during
  repository maintenance.
