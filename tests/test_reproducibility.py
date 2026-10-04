"""Reproducibility: the same input + seed gives identical metrics."""
import json
import shutil
import subprocess
import sys

from conftest import ROOT


def test_second_dataset_is_reproducible(retail_run):
    first = json.load(open(retail_run / "metrics.json"))
    run = "pytest_repro"
    try:
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.pipeline", "--config", "configs/retail_style_fixture.yaml", "--run", run],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        second = json.load(open(ROOT / "data" / "runs" / run / "processed" / "metrics.json"))
        assert first == second
    finally:
        shutil.rmtree(ROOT / "data" / "runs" / run, ignore_errors=True)


def test_fixture_generator_is_deterministic(tmp_path):
    import hashlib
    h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    p = ROOT / "examples" / "retail_style_fixture" / "transactions.csv.gz"
    import pandas as pd
    a = pd.read_csv(p)
    subprocess.run([sys.executable, "scripts/make_retail_fixture.py"], cwd=ROOT, check=True, capture_output=True)
    b = pd.read_csv(p)
    pd.testing.assert_frame_equal(a, b)
