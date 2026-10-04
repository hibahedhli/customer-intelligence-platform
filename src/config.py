"""
Central configuration: paths, random seed and business defaults.

Run layout
----------
* default run (``CIP_RUN`` unset)  -> data/standardized, data/processed, models, reports/figures
* named run  (``CIP_RUN=my_data``) -> data/runs/my_data/{standardized,processed,models,figures}

A run is selected with the ``CIP_RUN`` environment variable, with ``set_run()`` (used by the CLI
``--run`` flag), or in the dashboard sidebar. Modules must read paths as ``config.DATA_PROC`` etc.
at call time (not ``from .config import DATA_PROC``) so that ``set_run`` takes effect.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# --- only used by the synthetic data generator (NOT by the pipeline) ----------
SIM_N_CUSTOMERS = 6000
SIM_START = "2024-01-01"
SIM_END = "2025-12-31"
DATA_RAW = ROOT / "data" / "raw"          # where the synthetic generator writes its raw files

# --- defaults that dataset configs may override (see configs/*.yaml) ----------
GROSS_MARGIN = 0.30             # assumed share of revenue that is margin (CLV proxy)
HIGH_RISK_MIN_LIFT = 2.0        # "High" tier: churn rate of the flagged group >= 2 x base rate (validation)
LOW_RISK_MAX_CHURN = 0.15       # "Low" tier: churn rate of the group <= 15% (validation)

RUN = None
DATA_STD = DATA_PROC = MODELS = FIGURES = None
REPORTS = ROOT / "reports"


def paths_for(run=None) -> dict:
    """Directories of a run. ``None`` / ``'default'`` = the legacy single-dataset layout."""
    if run in (None, "", "default"):
        return dict(std=ROOT / "data" / "standardized", processed=ROOT / "data" / "processed",
                    models=ROOT / "models", figures=ROOT / "reports" / "figures")
    base = ROOT / "data" / "runs" / str(run)
    return dict(std=base / "standardized", processed=base / "processed", models=base / "models", figures=base / "figures")


def set_run(run=None) -> None:
    """Select the active run (also exported in CIP_RUN for child processes)."""
    global RUN, DATA_STD, DATA_PROC, MODELS, FIGURES
    run = None if run in (None, "", "default") else str(run)
    p = paths_for(run)
    RUN, DATA_STD, DATA_PROC, MODELS, FIGURES = run, p["std"], p["processed"], p["models"], p["figures"]
    if run is None:
        os.environ.pop("CIP_RUN", None)
    else:
        os.environ["CIP_RUN"] = run


def list_runs() -> list:
    """Runs that already have results: 'default' first, then data/runs/*."""
    out = []
    if (paths_for(None)["processed"] / "metrics.json").exists():
        out.append("default")
    root = ROOT / "data" / "runs"
    if root.exists():
        out += sorted(d.name for d in root.iterdir() if (d / "processed" / "metrics.json").exists())
    return out


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_STD, DATA_PROC, MODELS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)


set_run(os.environ.get("CIP_RUN") or None)
