"""Optional conversion and parity check in a separate MLX Python environment."""

import json
import math
import sys

from .store import absolute, atomic_json


def compare(reference, actual, tolerance=0.03):
    maximum = 0.0
    same = True
    if len(reference) != len(actual):
        return {"performed": True, "passed": False, "reason": "count mismatch"}
    for left, right in zip(reference, actual):
        if set(left) != set(right):
            return {"performed": True, "passed": False, "reason": "question mismatch"}
        for q, a in left.items():
            b = right[q]
            if "choice" in a:
                same &= a["choice"] == b.get("choice")
                pa, pb = a.get("probabilities", {}), b.get("probabilities", {})
                if set(pa) != set(pb):
                    same = False
                for key, value in pa.items():
                    difference = abs(float(value) - float(pb.get(key, float("nan"))))
                    if not math.isfinite(difference):
                        same = False
                    else:
                        maximum = max(maximum, difference)
            elif "noul" in a:
                difference = abs(float(a["noul"]) - float(b.get("noul", float("nan"))))
                if not math.isfinite(difference):
                    same = False
                else:
                    maximum = max(maximum, difference)
                same &= (a["noul"] >= 0.5) == (b.get("noul", 0) >= 0.5)
            else:
                same = False
    return {
        "performed": True,
        "passed": bool(same and maximum <= tolerance),
        "max_probability_difference": maximum,
        "tolerance": tolerance,
    }


def main():
    import laya_mlx
    from laya_mlx.convert import convert

    out = absolute(sys.argv[1])
    convert(str(out / "model"), str(out / "mlx"))
    data = json.loads((out / "parity-input.json").read_text())
    agent = laya_mlx.load(str(out / "mlx"))
    answers = [agent.predict(r["state"], r["questions"])["answers"] for r in data["rows"]]
    atomic_json(out / "parity.json", compare(data["reference"], answers))


if __name__ == "__main__":
    main()
