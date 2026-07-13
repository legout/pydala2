# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

## Repo notes

- The `wontfix` label **already exists** in `legout/pydala2` — do not recreate it.
- `needs-triage`, `needs-info`, `ready-for-agent`, and `ready-for-human` do **not** exist yet. Create them on first use (e.g. `gh label create needs-triage --color FBCA04`).
- This repo also uses the standard GitHub labels `bug`, `documentation`, `enhancement`, `good first issue`, `help wanted`, `question`, `duplicate`, and `invalid`. These are orthogonal to the triage state machine.

Edit the right-hand column to match whatever vocabulary you actually use.
