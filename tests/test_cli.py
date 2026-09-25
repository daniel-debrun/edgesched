from __future__ import annotations

import json

import pytest

from edgesched.cli import main

from .conftest import WORKLOADS


def test_analyze_exit_codes(capsys):
    assert main(["analyze", str(WORKLOADS / "multi_agent_cell.yaml")]) == 0
    assert "ADMITTED" in capsys.readouterr().out
    assert main(["analyze", str(WORKLOADS / "vla_plus_control.yaml")]) == 2
    assert "blocking" in capsys.readouterr().out
    path = str(WORKLOADS / "multi_agent_cell.yaml")
    assert main(["analyze", path, "--batching", "same_deadline"]) == 0
    assert "charged by the test: 0.455" in capsys.readouterr().out


def test_simulate_prints_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "s.json"
    args = ["simulate", str(WORKLOADS / "multi_agent_cell.yaml"), "-p", "edf_batch", "-p", "fifo"]
    assert main([*args, "--duration", "2", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "policy: edf_batch" in text and "policy: fifo" in text
    data = json.loads(out.read_text())
    assert set(data) == {"edf_batch", "fifo"}


def test_simulate_prometheus(capsys):
    args = ["simulate", str(WORKLOADS / "vla_plus_control.yaml"), "-p", "edf", "--duration", "1"]
    assert main([*args, "--prometheus"]) == 0
    assert "edgesched_deadline_miss_ratio" in capsys.readouterr().out


def test_bench_small_sweep(tmp_path, capsys):
    args = [
        "bench",
        str(WORKLOADS / "multi_agent_cell.yaml"),
        "--policies",
        "edf",
        "edf_batch",
        "--util-min",
        "0.5",
        "--util-max",
        "0.9",
        "--util-step",
        "0.4",
        "--seeds",
        "1",
        "--duration",
        "1",
        "--out-dir",
        str(tmp_path / "res"),
        "--figure-dir",
        str(tmp_path / "fig"),
    ]
    assert main(args) == 0
    assert "| 0.50 |" in capsys.readouterr().out
    assert (tmp_path / "res" / "multi_agent_cell_table.md").exists()
    assert (tmp_path / "fig" / "multi_agent_cell_miss_vs_util.png").stat().st_size > 1000


def test_profile_command_with_torch(tmp_path, capsys):
    pytest.importorskip("torch")
    out = tmp_path / "actor.json"
    args = [
        "profile",
        "--target",
        "edgesched.zoo:actor_mlp",
        "--batch-sizes",
        "1,4",
        "--iters",
        "3",
        "--warmup",
        "1",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    data = json.loads(out.read_text())
    assert set(data["points"]) == {"1", "4"}
    assert "fit" in capsys.readouterr().out


def test_unknown_policy_exits():
    with pytest.raises(SystemExit):
        main(["simulate", str(WORKLOADS / "synthetic_mix.yaml"), "-p", "lottery"])
