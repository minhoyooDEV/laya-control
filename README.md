# laya-control

Local, auditable multi-session dataset and training coordination for
[Laya](https://github.com/NandhaKishorM/laya). Independent community tooling.
Python 3.12+, macOS/Linux, local filesystem; no controller runtime dependencies.

## Features

- Concurrent submissions, provenance, exact deduplication and explicit label review.
- Authorized snapshots with stable source-group splits and a sealed benchmark.
- Finite job queues, one accelerator job per host, child-lifetime locks and cancellation.
- Independent candidates, evaluation gates, coverage checks and generation CAS.
- Immutable model/schema releases, live health verification and rollback.
- Private Unix inference and read-only JSON status for agents and menu bars.

Data preparation is concurrent. GPU execution is serialized in v0.1. Independent
weights are not combined: request a cumulative data union explicitly. Training
inclusion does not guarantee accuracy. No continuous learning is enabled.

## Install

```sh
python3 -m venv /tmp/laya-control-env
/tmp/laya-control-env/bin/pip install git+https://github.com/minhoyooDEV/laya-control.git
/tmp/laya-control-env/bin/laya-control status --json
```

Install this package into each training/serving environment. The optional adapter
requires `laya==0.3.28` and PyTorch; MLX conversion requires `laya-mlx` in its own
environment. Download models explicitly before running jobs.

State defaults to `~/.local/state/laya-control`. Set `LAYA_CONTROL_HOME` or absolute
`--state-root` outside repositories and synced/network drives. All workers share
the host accelerator lock at `~/.local/state/laya-control-locks`.

## Workflow

See [protocol](docs/protocol.md) for JSON formats. Paths below are illustrative.

```sh
laya-control submit --file /tmp/batch.json --request-id session-a-batch-1
laya-control confirm --file /tmp/review.json
laya-control authorize --file /tmp/review.json
laya-control snapshot --authorization-id AUTH_ID --base-revision BASE_TREE_SHA256
laya-control train --snapshot SNAPSHOT_ID --base /tmp/cached-base --python /tmp/training-env/bin/python --config /tmp/policy.json --owner session-a --request-id train-request-1
laya-control worker --drain
laya-control status --json
laya-control train --remote --snapshot SNAPSHOT_ID --config /tmp/policy.json --owner session-a --request-id train-request-2
laya-control status --remote
```

`train` queues; `worker` executes. Drain captures already authorized queued jobs.
Later submissions never change snapshots or create new jobs. 100 examples from A
plus 80 from B with 20 identical examples become 160 unique revisions. Another 15
after cutoff stay pending until a new explicit union authorization. Counts include
evaluation splits, not just optimizer inputs. Conflicting labels require review.

```sh
laya-control activate --run-id RUN_ID --expected-generation 0 --reason 'Apply reviewed candidate' --python /tmp/inference-env/bin/python --socket /tmp/lc-demo.sock
laya-control rollback --release-id PRIOR_RELEASE --expected-generation 2 --reason 'Restore prior behavior' --python /tmp/inference-env/bin/python --socket /tmp/lc-demo.sock
```

Activation is explicit, briefly interrupts serving, checks live release identity,
and restores the previous release on startup failure. Stale generations and lost
coverage are rejected. Rollback creates a new generation. Use one socket per task
domain. Never delete locks to force recovery. Use `recover` after worker failure
and `recover-activation` after an interrupted switch. Recovery never retrains.

## Privacy

This repository contains code and synthetic fixtures only. Keep datasets, models,
logs, databases, credentials and deployment configuration outside the repository.
There is no telemetry or upload implementation. Subprocesses must be trusted:
offline Hugging Face settings are not a firewall. Cloud agents can transmit records
shown in their context even when Laya runs locally. Ignore rules alone do not prove
publication safety.

## Validation and limitations

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q laya_control
```

Tests use real processes and synthetic data for concurrent submissions, worker
death, inherited locks and socket contention. They do not claim model accuracy.
The built-in adapter supports choice and boolean (`noul`) decisions. Score tasks
need another audited adapter. Calibration uses a separate split; baseline/candidate
per-class recall gates and optional PyTorch/MLX parity precede activation.
Insufficient calibration cannot pass. Unsupported Laya versions fail closed.

Exact deduplication does not detect semantic near-duplicates: review source groups
before authorization. Corrected active labels currently require explicit domain
migration or rollback. Custom commands are trusted adapter escape hatches, not
training-quality attestations. See [architecture](docs/architecture.md).
