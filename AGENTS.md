## Agent skills

### Issue tracker

Issues live as GitHub issues in `legout/pydala2` (use the `gh` CLI, which infers the repo from the remote). External pull requests are **not** treated as a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five canonical triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) with their default strings. The `wontfix` label already exists in this repo; the other four are created on first use. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root, created lazily by `/domain-modeling`. See `docs/agents/domain.md`.
