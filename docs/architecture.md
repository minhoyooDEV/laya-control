# Architecture and recovery

SQLite WAL serializes metadata writes on a local filesystem. Snapshot files are
fsynced and atomically published before DB commit. A pre-commit crash leaves an
orphan reported by reconcile, never deleted. Back up with SQLite's backup API.

Source groups receive persistent splits. The first test benchmark is sealed;
later test-assigned examples are future holdout. Authorization fixes exact reviewed
membership, never an all-future-records subscription.

Workers inherit an OS lock into the trainer child so supervisor death cannot
release the slot while the trainer lives. Custom adapters must preserve this FD.
Queued/running/evaluating/succeeded/failed/cancelled/interrupted stay distinct.
Cancellation checks process identity. Timeout never authorizes PID killing or lock
stealing. CPU-only jobs share the conservative slot in this release.

Releases copy checked models with schema and lineage. Activation requires expected
generation and coverage superset. Independent weights are not merged. Filesystem
and SQLite are not one transaction: a prepared/switched/verified journal supports
restoring the previous release after interruption. Live health is checked before
active is committed. Failed restoration is explicit.

The server owns its socket lifetime lock and refuses to unlink answering unmanaged
listeners. Model/schema load once. Short activation downtime avoids assuming memory
for two models. This is local development tooling, not an HA or distributed scheduler.
Dataset inclusion is auditable; accuracy and GPU capacity need task measurements.
