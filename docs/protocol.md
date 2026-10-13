# Local protocol

Submission files contain arrays. This example is synthetic:

```json
[{"task_id":"colors","schema":{"color":{"type":"choice","instructions":"Choose the stated color","criteria":["red","blue"]}},"state":"The card is red.","expected":{"color":"red"},"source_group":"card-001","session_id":"session-a","origin":"synthetic"}]
```

Origins: user, authoritative_source, rule_batch, synthetic, pseudo. Extra source
references and rule versions are preserved. Pseudo labels are train-only. A request
ID replay with changed content fails. Different labels create unresolved conflicts.

Review and authorization are separate operations using this file shape:

```json
{"revision_ids":["REVISION_ID"],"actor":"reviewer","reason":"Reviewed using the labeling rule"}
```

`confirm` records review; `authorize` records the actual user's training request
and freezes membership. Agents must not invent consent. This is an audit trail,
not access control against another process under the same OS user.
`resolve --file` accepts example_key, revision_id, actor, reason.

## Training

The base is an absolute local directory. Its identity is SHA256 of canonical JSON
mapping relative file names to SHA256 of contents, including cached file targets.
Canonical JSON uses sorted keys, comma/colon separators, UTF-8, no ASCII escaping.
Do not modify a base during training. The adapter verifies it before loading.

```json
{"device":"mps","train":{"epochs":1,"micro_batch":1,"grad_accum":4,"freeze_encoder":true},"policy":{"min_accuracy":0.8,"max_regression":0.0},"mlx_python":"/tmp/inference-env/bin/python"}
```

Omit mlx_python for PyTorch output. Policy is task-specific. Config is copied into
content-addressed local state when queued.

Custom `enqueue --config` accepts a trusted argv array:

```json
{"command":["/absolute/python","/absolute/adapter.py","{snapshot}","{run_dir}"]}
```

No shell interpolation. Adapters must preserve inherited LAYA_LOCK_FD until all
accelerator work ends. Early close is unsupported. Write isolated model/ or mlx/
and result.json after actual training and evaluation:

```json
{"trained":true,"runtime":"pytorch","model_subdir":"model","evaluation":{"passed":true}}
```

An adapter must provide meaningful evidence. The controller verifies exit status,
artifacts and lineage but cannot attest arbitrary executable behavior. Use the
built-in adapter for its documented evaluation checks.

## Remote training

`train --remote` sends the snapshot's four splits to a shared Laya server instead of
queuing a local job. The client library comes from a laya-poc checkout: set
`LAYA_POC_HOME` (default `~/Documents/laya-poc`, `client/` must exist) and
`LAYA_SERVER_URL` (default `http://127.0.0.1:7650`).

```sh
laya-control train --remote --snapshot SNAPSHOT_ID --config /tmp/policy.json --owner session-a --request-id train-request-1
laya-control status --remote
laya-control activate --remote --run-id REMOTE_JOB_ID --reason 'Apply reviewed candidate'
```

Only task `compact-laya-focus-v2` is supported. The command refuses to upload when the
server's `/v1/status` `base.files` digest differs from the snapshot `base_revision`, or
when the registered task schema differs from the snapshot schema. From the config
`train` object only `epochs`, `micro_batch`, `grad_accum`, `freeze_encoder` and `seed`
are sent; other keys are reported as `ignored_train_keys`. `device`, `mlx_python` and
`--python`/`--base` are not used. `policy` is frozen into the job record at submit time.

`activate --remote` takes the finished job's server metrics and gates them against the
base model, on dev and then, only if dev passed, on test. A dev-rejected job's test metrics
are never read. The rule differs from local runs: pooled accuracy must reach `min_accuracy`,
and every question's balanced accuracy (mean recall of its classes) must be at least the
base's minus `max_regression`. A per-class recall rule would block any model that learns to
say "no" when the base nearly always says "yes". A binary question with a class of fewer than
5 examples in the split is compared by plain accuracy instead (`candidate >= base - max_regression`)
and gets the warning `<qid>: small class support (<5), compared by accuracy`. The server sends no
per-class counts, so support is recovered from the integer hit counts that fit the rounded
accuracy and recalls of both base and candidate (a `support` field is used if the server adds
one, and a recall dict missing a class counts as small). Numbers that fit no integer counts
stay on the balanced comparison. Parity must pass. A fitted temperature on the
`[0.5, 5]` clamp, or `temperature_at_bound`, is recorded as a warning; any other calibration
issue rejects. Only a passing job is activated on the server. The decision (balanced accuracies
per question for candidate and base, reasons, warnings) is stored in the job record and printed
under `evaluation`. Server-side job data is not deleted: the server keeps a cumulative dataset.
Generation and data-coverage checks are local-release features and do not apply.

A request ID is recorded before the upload. If the process dies mid-upload, the same
request ID is refused until the server job list is checked; use a new request ID to
submit again. Remote job records live under `<state-root>/remote/`.

## Serving and status

One JSON request per Unix connection; 1 MiB limit; 5-second input timeout.
`{"kind":"health"}` returns ready, release_id, generation, snapshot_id.
`{"kind":"predict","state":"text"}` uses the frozen release schema.

`status --json` initializes nothing, starts nothing and excludes raw examples.
pending_count means not yet in a snapshot, not necessarily unlearned. Also inspect
job states and active lineage. Failed runs require a new explicit queue operation.
