import csv

import numpy as np
import pytest
import yaml

from tan import runner

SPEC = {
    "name": "t_runner",
    "runs": 2,
    "regions": ["synthetic"],
    "set": {
        "simulation.duration": 60.0,
        "simulation.flight.margin_sigma0": 500.0,
        "particle_filter.n_particles": 300,
    },
    "filters": {
        "dead_reckoning": {"family": "dead_reckoning"},
        "ekf": {"family": "ekf"},
        "pf_small": {"family": "particle_filter", "set": {"particle_filter.n_particles": 300}},
    },
    "grid": {"init.position_sigma": [100.0, 300.0]},
}


@pytest.fixture
def exp_dirs(tmp_path, monkeypatch):
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "t_runner.yaml").write_text(yaml.safe_dump(SPEC))
    (exp / "_common.yaml").write_text(yaml.safe_dump({"set": {"simulation.flight.mode": "random"}}))
    monkeypatch.setattr(runner, "EXPERIMENTS_DIR", exp)
    monkeypatch.setattr(runner, "COMMON_FILE", exp / "_common.yaml")
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


def test_expand_grid():
    configs = runner.expand(SPEC)
    assert len(configs) == 2
    assert configs[0].cfg["init"]["position_sigma"] == 100.0
    assert configs[1].id == "synthetic__init.position_sigma=300"


def test_sweep_writes_resumes_and_keeps_crn(exp_dirs, capsys):
    out = runner.sweep("t_runner", jobs=2)
    files = sorted(out.rglob("run_*.npz"))
    assert len(files) == 4

    a = np.load(out / "main" / "synthetic__init.position_sigma=100" / "run_0001.npz")
    b = np.load(out / "main" / "synthetic__init.position_sigma=300" / "run_0001.npz")
    np.testing.assert_array_equal(a["truth"], b["truth"])  # same flight in both configs
    assert a["pf_small__est"].shape == (60, 4) and a["pf_small__cov"].shape == (60, 10)
    assert "pf_small__ess" in a and "ekf__ess" not in a

    with open(out / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 3
    assert {r["filter"] for r in rows} == {"dead_reckoning", "ekf", "pf_small"}
    assert {r["init.position_sigma"] for r in rows} == {"100.0", "300.0"}

    mtime = files[0].stat().st_mtime
    capsys.readouterr()
    runner.sweep("t_runner", jobs=2)
    assert "0 to run" in capsys.readouterr().out
    assert files[0].stat().st_mtime == mtime


def test_unpack_cov_roundtrip(rng):
    a = rng.standard_normal((5, 4, 4))
    cov = a @ np.swapaxes(a, 1, 2)
    upper = cov[:, runner._UPPER[0], runner._UPPER[1]]
    np.testing.assert_allclose(runner.unpack_cov(upper), cov)
