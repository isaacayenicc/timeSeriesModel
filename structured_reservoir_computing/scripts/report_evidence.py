"""Read saved SRC evidence for report and presentation generation."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START, STOP = 1792, 24000


def number(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            return num
    return None


def fmt(row, *keys, digits=6, missing="—"):
    value = number(row, *keys)
    return missing if value is None else f"{value:,.{digits}f}"


def name(row):
    key = model_key(row)
    common = {
        "primary-pi-src": "Selected PI-SRC",
        "pi-src-adapter-off": "PI-SRC, same weights without adapter",
        "archived-src-corrected": "Archived corrected SRC",
        "selected-pi-src": "Selected PI-SRC",
        "strict-sparse-lpv-arma": "Strict sparse LPV–ARMA",
        "polynomial-src": "Polynomial SRC",
        "pi-src-fixed": "PI-SRC-Fixed",
        "pi-src-optimized": "PI-SRC-Optimized (uncorrected)",
        "src-generic": "SRC-Generic",
        "src-polynomial": "SRC-Polynomial",
        "pi-src-no-angle": "PI-SRC without angle lags",
        "pi-src-no-carrier": "PI-SRC without oscillator bank",
        "pi-src-no-innovation": "PI-SRC without innovations",
        "pi-src-no-augmentation": "PI-SRC without augmentation",
        "pi-src-degree2": "PI-SRC with degree two",
        "pi-src-order12": "PI-SRC with 12 orders",
        "pi-src-fixed-hz-only": "PI-SRC with fixed-Hz states only",
        "sparse-lpv-arma": "Sparse LPV–ARMA",
        "sparse-lpv-ar": "Sparse LPV–AR",
        "raw": "Raw vibration",
        "cnn": "CNN",
        "ridge": "Ridge",
    }
    return common.get(key) or row.get("display") or row.get("name") or str(row.get("model", "—")).replace("_", " ")


def model_key(row):
    return str(row.get("model", row.get("model_id", ""))).lower().replace("_", "-").replace(" ", "-")


@dataclass(frozen=True)
class Evidence:
    root: Path = ROOT

    def csv(self, *names):
        for relative in names:
            path = self.root / relative
            if path.is_file():
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    return list(csv.DictReader(stream))
        return []

    def json(self, *names):
        for relative in names:
            path = self.root / relative
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def plot(self, *names):
        for relative in names:
            path = self.root / "figures" / relative
            if path.is_file():
                return path
        return None

    @property
    def comparison(self):
        return self.csv("results/metrics/all_model_comparison.csv")

    @property
    def measured(self):
        return [row for row in self.comparison if number(row, "mse") is not None]

    @property
    def selection(self):
        return self.json("results/metrics/src_summary.json")

    @property
    def primary(self):
        selected = self.selection
        key = model_key(selected)
        for row in self.measured:
            if model_key(row) == key and key:
                return row
        for row in self.measured:
            if model_key(row) in ("pi-src", "src-primary", "primary-src", "selected-src", "primary-pi-src"):
                return row
        return {}

    @property
    def sparse_reference(self):
        for row in self.measured:
            key = model_key(row) + name(row).lower()
            if "lpv" in key and "arma" in key and ("sparse" in key or "spa-" in key) and "archive" not in key:
                return row
        return {}

    @property
    def archive_reference(self):
        candidates = [row for row in self.measured if "archive" in model_key(row)
                      and ("src" in model_key(row) or "src" in name(row).lower())]
        runnable = [row for row in candidates if "checkpoint" in str(row.get("scope", "")).lower()]
        return min(runnable, key=lambda row: number(row, "mse"), default={})

    @property
    def comparison_top(self):
        ranked = sorted(self.measured, key=lambda row: number(row, "mse"))
        chosen = [row for row in (self.primary, self.sparse_reference, self.archive_reference) if row]
        baselines = [row for row in ranked if row.get("scope") == "replayed comparator"]
        neural = next((row for row in baselines if model_key(row) in {"mlp", "lstm", "gru", "cnn", "tcn", "transformer", "tft"}), {})
        for row in [*baselines[:2], neural, *ranked]:
            if row and row not in chosen and len(chosen) < 6:
                chosen.append(row)
        return sorted(chosen, key=lambda row: number(row, "mse"))

    @property
    def ablations(self):
        return self.csv("results/ablations/src_ablations.csv")

    @property
    def fault(self):
        return self.csv("results/fault/fault_group_summary.csv",
                        "results/fault/src_live/fault_group_summary.csv",
                        "results/fault/comparator_live/fault_group_summary.csv")

    @property
    def stress(self):
        return self.csv("results/fault/rpm_stress_summary.csv",
                        "results/fault/src_live/rpm_stress_summary.csv",
                        "results/fault/comparator_live/rpm_stress_summary.csv")

    @property
    def pulses(self):
        return self.csv("results/fault/pulse_summary.csv",
                        "results/fault/src_live/pulse_summary.csv",
                        "results/fault/comparator_live/pulse_summary.csv")

    @property
    def eda(self):
        return self.json("results/eda/summary.json", "results/eda/eda_summary.json")

    def comparison_text(self):
        mse = number(self.primary, "mse")
        if mse is None:
            return "A matched primary development score is unavailable in the saved evidence."
        reference = number(self.sparse_reference, "mse")
        if reference is None or reference == 0:
            return f"PI-SRC development MSE {mse:.6f} on the common target mask."
        improvement = 100 * (reference - mse) / reference
        direction = "lower" if improvement >= 0 else "higher"
        return f"PI-SRC MSE {mse:.6f} is {abs(improvement):.2f}% {direction} than the strict sparse LPV–ARMA result."
