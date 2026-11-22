"""Plot independently measured full-window overlap statistics."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_overlap(rows_path: Path, dest: Path):
    """Expose possible leakage without showing or copying the source waveforms."""
    with Path(rows_path).open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["record_id"].startswith("sigfault1:")]
    if not rows:
        return False
    figure, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8.5, 4.4), layout="constrained")
    for ax, key, color in zip(axs, ("sigtrain", "sigvalid"), ("#0A6C75", "#345790"), strict=True):
        subset = [row for row in rows if row["reference"] == key]
        ids = [int(row["record_id"].split(":")[-1]) for row in subset]
        corr = [float(row["x_correlation"]) for row in subset]
        ax.scatter(ids, corr, c=color, s=12, alpha=.82)
        for i, value, row in zip(ids, corr, subset, strict=True):
            if row["exact_full_length"] == "True":
                ax.scatter([i], [value], c="#A63F16", s=35, zorder=3)
                ax.annotate(f"Record {i}\nExact x and RPM", xy=(i, value), xytext=(i+13, .75),
                            arrowprops={"arrowstyle": "-", "color": "#A63F16"}, color="#833B20")
        ax.set(ylabel=f"{key}\nPeak x correlation", ylim=(0,1.08))
        ax.set_yticks(np.arange(0,1.1,.25))
        ax.grid(axis="y", color="#DFE9EA")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axs[-1].set(xlabel="One-based course diagnostic record number", xlim=(0,max(ids)+1))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True
