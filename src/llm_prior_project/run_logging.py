from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_INDEX_FIELDS = [
    "timestamp",
    "dataset",
    "experiment",
    "tag",
    "run_dir",

    "provider_model",
    "prompt_version",
    "prompt_single_sha256",
    "prompt_sampling_sha256",

    "split",
    "alpha",
    "seed",
    "num_samples",

    "n_train",
    "n_test",

    "r2_baseline",
    "rmse_baseline",
    "r2_tir",
    "rmse_tir",
    "r2_cov",
    "rmse_cov",

    "delta_r2_tir_vs_base",
    "delta_r2_cov_vs_base",

    "config_sha256",
]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Dict[str, Any]) -> str:
    # stable hashing: sort keys + compact separators
    txt = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(txt)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def create_run_dir(root: Path, dataset: str, experiment: str, tag: str) -> Path:
    """
    Creates:
      <root>/<dataset>/<experiment>/<timestamp>_<tag>/
        figures/
    Never overwrites (errors if exists).
    """
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = root / dataset / experiment / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def append_index_row(index_csv: Path, row: Dict[str, Any], field_order: Optional[list[str]] = None) -> None:
    """
    Appends a single row to runs/locked/index.csv.
    Creates file with header if missing or empty.
    Missing columns are written as empty.
    Extra keys not in field_order are appended at the end (stable).
    """
    ensure_parent(index_csv)

    # Determine column order
    base_fields = field_order[:] if field_order else DEFAULT_INDEX_FIELDS[:]
    extra_fields = [k for k in row.keys() if k not in base_fields]
    fields = base_fields + sorted(extra_fields)

    file_exists = index_csv.exists()
    needs_header = (not file_exists) or (index_csv.stat().st_size == 0)

    with index_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            writer.writeheader()

        # Fill missing
        out = {k: row.get(k, "") for k in fields}
        writer.writerow(out)
