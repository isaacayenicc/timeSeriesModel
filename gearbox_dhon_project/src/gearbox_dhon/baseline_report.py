"""Build a four-method reconciliation from the conventional experiment files."""
from __future__ import annotations

import csv
from pathlib import Path

from gearbox_dhon.baselines import PUBLISHED, write_rows

RETAINED = {"ari": 275, "arima": 290, "spa_lpv_ar": 1205, "spa_lpv_arma": 593}
ARCHIVE_RETAINED = {"ari": 275, "arima": 290, "spa_lpv_ar": 1289, "spa_lpv_arma": 636}


def _read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def reconciliation(run: Path):
    """Collect original, rerun and common-protocol values without joining unequal masks."""
    metrics = run / "metrics"
    if not (metrics / "conventional_metrics.csv").is_file():
        return
    measurements = {r["model"]: r for r in _read(metrics / "conventional_metrics.csv")}
    if not all(f"archived_{name}" in measurements for name in PUBLISHED):
        return
    modes = {}
    for basis in ("legendre", "monomial"):
        path = metrics / f"paper_convention_{basis}.csv"
        search = run / f"search/lpv_paper_row_cv_{basis}.csv"
        if path.is_file() and search.is_file():
            rows = _read(path)
            fits = {r["stage"]: r for r in _read(search) if r["fold"] == "full"}
            modes[basis] = {r["model"]: (r, fits[str(k + 1)]) for k, r in enumerate(rows)}
    rows = []
    for name, (train, test, runs) in PUBLISHED.items():
        archived = measurements[f"archived_{name}"]
        strict = measurements[name]
        row = dict(model=name, paper_train_mse=train, paper_validation_mse=test, paper_validation_runs_p=runs,
                   paper_retained_stage_coefficients=RETAINED[name],
                   archive_train_mse=archived["historical_train_mse"],
                   archive_validation_mse=archived["historical_mse"],
                   archive_validation_runs_p=archived["historical_runs_p"],
                   archive_validation_start=archived["historical_start"],
                   archive_retained_stage_coefficients=ARCHIVE_RETAINED[name],
                   archive_common_mse=archived["mse"], strict_train_mse=strict["historical_train_mse"],
                   strict_validation_mse=strict["mse"], strict_validation_runs_p=strict["runs_p"],
                   strict_common_start=strict["first_index"])
        for basis, candidates in modes.items():
            prediction, fit = candidates.get(name, ({}, {}))
            row.update({
                f"{basis}_random_row_train_mse": fit.get("mse"),
                f"{basis}_random_row_validation_mse": prediction.get("mse"),
                f"{basis}_random_row_runs_p": prediction.get("runs_p"),
                f"{basis}_random_row_start": prediction.get("start"),
                f"{basis}_random_row_retained_stage_coefficients": fit.get("support"),
            })
        rows.append(row)
    write_rows(run / "tables/lpv_paper_comparison.csv", rows)
