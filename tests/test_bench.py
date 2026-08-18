from marathon.bench import main, simulate


def test_simulation_saves_bytes():
    report = simulate(turns=20, growth=300, seed=3)
    totals = report["totals"]
    assert totals["wire_bytes"] < totals["full_bytes"] / 3
    assert 0 < totals["savings_ratio"] < 1
    assert len(report["turns"]) == 20


def test_simulation_with_edits_still_saves():
    report = simulate(turns=30, growth=300, edit_every=5, seed=3)
    assert report["totals"]["savings_ratio"] > 0.5


def test_simulation_is_deterministic():
    a = simulate(turns=10, growth=200, seed=42)
    b = simulate(turns=10, growth=200, seed=42)
    assert a == b


def test_cli_runs_and_writes_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["--turns", "5", "--growth", "100", "--json", str(out)]) == 0
    assert out.exists()
    captured = capsys.readouterr()
    assert "totals:" in captured.out
