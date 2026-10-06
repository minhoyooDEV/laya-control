"""Optional Laya 0.3.28 adapter. Imports model libraries only in an explicit job."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import subprocess

from .store import ControlError, absolute, atomic_json, digest


def rows_for(snapshot, split):
    ids = snapshot["splits"][split]
    if split == "test":
        if not set(snapshot["benchmark"]) <= set(ids):
            raise ControlError("sealed benchmark is incomplete")
        ids = list(snapshot["benchmark"])
    return [
        {
            "state": snapshot["examples"][rid]["state"],
            "questions": snapshot["examples"][rid]["schema"],
            "expected": snapshot["examples"][rid]["expected"],
        }
        for rid in ids
    ]


def score_predictions(rows, predictions):
    if len(rows) != len(predictions):
        raise ControlError("prediction count mismatch")
    counts = {}
    hits = {}
    total = correct = 0
    confusion = {}
    for row, answers in zip(rows, predictions):
        for q, question in row["questions"].items():
            expected = row["expected"][q]
            answer = answers[q]
            if question["type"] == "choice":
                predicted = answer["choice"]
            elif question["type"] == "noul":
                value = float(answer["noul"])
                if not math.isfinite(value):
                    raise ControlError("non-finite output")
                predicted = value >= 0.5
            else:
                raise ControlError("built-in evaluation supports choice and noul only")
            key = f"{q}:{expected}"
            counts[key] = counts.get(key, 0) + 1
            ok = predicted == expected
            hits[key] = hits.get(key, 0) + int(ok)
            pair = f"{key}->{predicted}"
            confusion[pair] = confusion.get(pair, 0) + 1
            total += 1
            correct += ok
    return {
        "accuracy": correct / total if total else 0,
        "count": total,
        "per_class_recall": {k: hits[k] / v for k, v in counts.items()},
        "confusion": confusion,
    }


def passes_policy(baseline, candidate, policy):
    if not baseline["count"] or candidate["count"] != baseline["count"]:
        return False
    delta = policy.get("max_regression", 0)
    if candidate["accuracy"] < policy.get("min_accuracy", 0.8):
        return False
    if candidate["accuracy"] < baseline["accuracy"] - delta:
        return False
    return all(
        candidate["per_class_recall"].get(k, -1) >= v - delta for k, v in baseline["per_class_recall"].items()
    )


def predict_rows(agent, rows):
    return [agent.predict(row["state"], row["questions"])["answers"] for row in rows]


def run_child(command):
    fd = os.environ.get("LAYA_LOCK_FD")
    return subprocess.run(command, check=True, pass_fds=(int(fd),) if fd else ())


def run(snapshot_path, run_dir, base, config_path):
    if importlib.metadata.version("laya") != "0.3.28":
        raise ControlError(
            "this adapter is validated against laya==0.3.28; use a separate adapter for other versions"
        )
    import laya
    from laya import train as training

    snapshot = json.loads(absolute(snapshot_path).read_text())
    out = absolute(run_dir)
    config = json.loads(absolute(config_path).read_text())
    base = absolute(base)
    # The exact immutable local base is a caller-supplied SHA256 tree identity.
    files = {
        str(p.relative_to(base)): __import__("hashlib").sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }
    if digest(files) != snapshot["base_revision"]:
        raise ControlError("base checksum does not match authorization snapshot")
    policy = config.get("policy", {})
    device = config.get("device", "cpu")
    split_rows = {s: rows_for(snapshot, s) for s in ("train", "dev", "calibration", "test")}
    if any(not rows for rows in split_rows.values()):
        raise ControlError("all four confirmed splits are required")
    for rows in split_rows.values():
        for row in rows:
            if any(q["type"] not in ("choice", "noul") for q in row["questions"].values()):
                raise ControlError("built-in adapter supports choice/noul only")
    baseline = laya.load(str(base), device=device)
    baseline_dev = score_predictions(split_rows["dev"], predict_rows(baseline, split_rows["dev"]))
    del baseline
    gc.collect()
    model, tok, cfg = training.load_checkpoint(str(base))
    if cfg.get("option_layout") not in (None, "sequential"):
        raise ControlError("unsupported option layout")
    tc = training.TrainConfig(**config.get("train", {}))
    tc.validate()
    max_len = tc.max_len or cfg.get("max_len", 512)
    head_len = tc.head_max_len or cfg.get("head_max_len", 192)
    train_items, skipped = training.items_from_rows(tok, split_rows["train"], max_len, head_len)
    expected = sum(len(r["questions"]) for r in split_rows["train"])
    if skipped or len(train_items) != expected:
        raise ControlError("some training labels could not be encoded")
    dev = training.resolve_device(device)
    history = training.train_model(model, tok, train_items, tc, dev, max_len, head_len)
    if not history or any(not math.isfinite(x) for x in history):
        raise ControlError("non-finite training loss")
    cal_items, skipped = training.items_from_rows(tok, split_rows["calibration"], max_len, head_len)
    if skipped:
        raise ControlError("calibration labels could not be encoded")
    records = training.calibration_records(model, tok, cal_items, dev, max_len, head_len)
    fitted = training.fit_temperature_map(records)
    cfg = dict(cfg, temperature=fitted["temperature"], max_len=max_len, head_max_len=head_len)
    cfg.pop("temperature_by_options", None)
    if fitted.get("temperature_by_options"):
        cfg["temperature_by_options"] = fitted["temperature_by_options"]
    report = training.calibration_report(records, fitted["temperature"])
    training.save_checkpoint(model, tok, cfg, str(out / "model"))
    del model, tok
    gc.collect()
    candidate = laya.load(str(out / "model"), device=device)
    candidate_dev = score_predictions(split_rows["dev"], predict_rows(candidate, split_rows["dev"]))
    accepted = passes_policy(baseline_dev, candidate_dev, policy)
    candidate_test = None
    baseline_test = None
    # A dev-rejected candidate never sees the sealed benchmark.
    if accepted:
        candidate_test = score_predictions(split_rows["test"], predict_rows(candidate, split_rows["test"]))
    parity_rows = split_rows["dev"]
    reference = predict_rows(candidate, parity_rows)
    del candidate
    gc.collect()
    if accepted:
        baseline = laya.load(str(base), device=device)
        baseline_test = score_predictions(split_rows["test"], predict_rows(baseline, split_rows["test"]))
        del baseline
        gc.collect()
        accepted = passes_policy(baseline_test, candidate_test, policy)
    runtime = "pytorch"
    parity = {"performed": False}
    if config.get("mlx_python"):
        atomic_json(out / "parity-input.json", {"rows": parity_rows, "reference": reference})
        run_child([str(absolute(config["mlx_python"])), "-m", "laya_control.mlx_step", str(out)])
        parity = json.loads((out / "parity.json").read_text())
        runtime = "mlx"
        accepted = accepted and parity["passed"]
    calibrated = all(not x["issues"] for x in report.values())
    atomic_json(
        out / "result.json",
        {
            "trained": True,
            "runtime": runtime,
            "model_subdir": "mlx" if runtime == "mlx" else "model",
            "consumed_items": len(train_items) * tc.epochs,
            "loss": history,
            "calibration": report,
            "evaluation": {
                "passed": bool(accepted and calibrated),
                "baseline_dev": baseline_dev,
                "candidate_dev": candidate_dev,
                "baseline_test": baseline_test,
                "candidate_test": candidate_test,
                "policy": policy,
            },
            "parity": parity,
            "versions": {"laya": importlib.metadata.version("laya")},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("snapshot", "run-dir", "base", "config"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    run(a.snapshot, a.run_dir, a.base, a.config)


if __name__ == "__main__":
    main()
