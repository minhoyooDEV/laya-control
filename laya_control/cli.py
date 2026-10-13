"""Explicit local operations; status never initializes state or starts services."""

import argparse
import json
import os
import sys
from pathlib import Path

from .store import ControlError, Store, absolute


def read(path):
    return json.loads(absolute(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-control")
    default = os.environ.get("LAYA_CONTROL_HOME", str(Path.home() / ".local/state/laya-control"))
    parser.add_argument("--state-root", default=default)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    p.add_argument("--remote", action="store_true")
    sub.add_parser("reconcile")
    sub.add_parser("recover")
    p = sub.add_parser("submit")
    p.add_argument("--file", required=True)
    p.add_argument("--request-id", required=True)
    for name in ("confirm", "resolve", "authorize"):
        sub.add_parser(name).add_argument("--file", required=True)
    p = sub.add_parser("snapshot")
    p.add_argument("--authorization-id", required=True)
    p.add_argument("--base-revision", required=True)
    p = sub.add_parser("enqueue")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--request-id", required=True)
    p = sub.add_parser("train")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--request-id", required=True)
    p.add_argument("--python")
    p.add_argument("--base")
    p.add_argument("--remote", action="store_true")
    p = sub.add_parser("worker")
    p.add_argument("--drain", action="store_true")
    p = sub.add_parser("cancel")
    p.add_argument("--run-id", required=True)
    for name in ("activate", "rollback", "recover-activation"):
        p = sub.add_parser(name)
        local_only = name != "activate"  # activate --remote needs none of the local serving arguments
        p.add_argument("--python", required=local_only)
        p.add_argument("--socket", required=local_only)
        if name != "recover-activation":
            p.add_argument("--run-id" if name == "activate" else "--release-id", required=True)
            p.add_argument("--expected-generation", type=int, required=local_only)
            p.add_argument("--reason", required=True)
        if name == "activate":
            p.add_argument("--remote", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = absolute(args.state_root)
        if args.action == "status":
            result = Store.read_status(root)
            if args.remote and result.get("state") == "configured":
                from .remote import make_remote, remote_status

                result["remote"] = remote_status(root, make_remote)
        else:
            store = Store(root)
            if args.action == "submit":
                rows = read(args.file)
                if not isinstance(rows, list):
                    raise ControlError("submission file must be a JSON array")
                result = [store.submit(row, f"{args.request_id}:{i}") for i, row in enumerate(rows)]
            elif args.action in ("confirm", "resolve", "authorize"):
                request = read(args.file)
                if args.action == "confirm":
                    store.confirm(request["revision_ids"], request["actor"], request["reason"])
                    result = {"confirmed": len(request["revision_ids"])}
                elif args.action == "resolve":
                    store.resolve(
                        request["example_key"], request["revision_id"], request["actor"], request["reason"]
                    )
                    result = {"resolved": True}
                else:
                    result = {
                        "authorization_id": store.authorize(
                            request["revision_ids"], request["actor"], request["reason"]
                        )
                    }
            elif args.action == "snapshot":
                snap = store.snapshot(args.authorization_id, args.base_revision)
                result = {k: v for k, v in snap.items() if k != "examples"}
            elif args.action == "train" and args.remote:
                from .remote import make_remote, submit_remote

                result = submit_remote(
                    store, make_remote(), args.snapshot, read(args.config), args.owner, args.request_id
                )
            elif args.action == "activate" and args.remote:
                from .remote import activate_remote, make_remote

                result = activate_remote(store, make_remote(), args.run_id, args.reason)
            elif args.action in ("enqueue", "train"):
                from .jobs import enqueue

                if args.action == "train":
                    if not args.python or not args.base:
                        raise ControlError("--python and --base are required unless --remote is given")
                    # Freeze the training policy alongside job state; later edits to the user's config cannot alter it.
                    from .store import atomic_json, digest

                    cfg = read(args.config)
                    path = store.root / "objects" / ("config-" + digest(cfg) + ".json")
                    atomic_json(path, cfg)
                    config = {
                        "command": [
                            str(absolute(args.python)),
                            "-m",
                            "laya_control.training",
                            "--snapshot",
                            "{snapshot}",
                            "--run-dir",
                            "{run_dir}",
                            "--config",
                            str(path),
                            "--base",
                            str(absolute(args.base)),
                        ]
                    }
                else:
                    config = read(args.config)
                config["request_id"] = args.request_id
                result = enqueue(store, args.snapshot, config, args.owner)
            elif args.action == "worker":
                from .jobs import work

                result = work(store, drain=args.drain)
            elif args.action == "cancel":
                from .jobs import cancel

                result = cancel(store, args.run_id)
            elif args.action == "recover":
                from .jobs import recover

                result = recover(store)
            elif args.action == "reconcile":
                result = store.reconcile()
            else:
                from .releases import activate, recover_activation, rollback
                from .server import ManagedService

                if args.action == "activate" and (
                    not args.python or not args.socket or args.expected_generation is None
                ):
                    raise ControlError(
                        "--python, --socket and --expected-generation are required unless --remote"
                    )
                service = ManagedService(args.socket, args.python)
                if args.action == "activate":
                    result = activate(store, args.run_id, args.expected_generation, args.reason, service)
                elif args.action == "rollback":
                    result = rollback(store, args.release_id, args.expected_generation, args.reason, service)
                else:
                    result = recover_activation(store, service)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ControlError, ValueError, KeyError, OSError) as exc:
        # Only our controlled diagnostics are printed; unexpected external messages can contain private input.
        message = str(exc) if isinstance(exc, ControlError) else type(exc).__name__
        print(json.dumps({"error": {"code": type(exc).__name__, "message": message}}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
