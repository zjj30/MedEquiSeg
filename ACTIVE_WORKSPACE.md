# Active workspace policy

The only writable remote workspace for this revision is:

`/tank2/zjj/MedEquiSeg_active`

The historical tree at `/tank2/zjj/project3` is read-only for this workflow.
Its datasets and pretrained model assets may be accessed through symbolic links,
but its source files, paper backups, logs, caches, checkpoints, and result tables
must not be copied into or used as evidence by the active revision.

## Writable locations

- Source, manuscript, and generated tables: the active Git working tree.
- New training checkpoints and caches: `outputs/` under the active tree.
- New run metadata and logs: `logs/` under the active tree.
- New compiled submission artifacts: `output/` under the active tree.

## Read-only external resources

- Public datasets: `/tank2/zjj/project3/datasets` via `datasets`.
- Pretrained model assets, when needed: `/tank2/zjj/project3/models` via `models`.

## Evidence boundary

Only files generated from the active branch and carrying the active protocol,
manifest, code, dependency, checkpoint, and seed provenance may update the paper.
The invalid `Shared plan (no rewrite)` configuration and historical 90/90
summary are excluded. Historical files may be consulted only to diagnose prior
claims or reproduce an audit finding.
