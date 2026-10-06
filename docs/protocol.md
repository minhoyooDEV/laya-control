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

## Serving and status

One JSON request per Unix connection; 1 MiB limit; 5-second input timeout.
`{"kind":"health"}` returns ready, release_id, generation, snapshot_id.
`{"kind":"predict","state":"text"}` uses the frozen release schema.

`status --json` initializes nothing, starts nothing and excludes raw examples.
pending_count means not yet in a snapshot, not necessarily unlearned. Also inspect
job states and active lineage. Failed runs require a new explicit queue operation.
