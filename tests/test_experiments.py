import math

import pytest

from run_experiments import parse, read_csv, rq1_figure, run, run_grid, summarise

TINY = ["--networks", "EB", "--n", "1000", "--replicates", "3", "--cv", "0", "1.5"]


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rows():
    return run_grid(parse([*TINY, "--workers", "2"]))


def test_grid_has_one_m1_row_per_replicate_and_one_m2_row_per_cell(rows):
    assert len(rows) == 2 * 3 * (1 + 2 * 3)
    assert [row["model"] for row in rows[:7]] == ["M1"] + ["M2"] * 6
    assert {(row["cv"], row["protocol"]) for row in rows if row["model"] == "M2"} == {
        (cv, protocol) for cv in (0.0, 1.5) for protocol in ("i", "ii", "iii")
    }


def test_the_grid_is_reproducible_from_its_seed_whatever_the_workers(rows):
    """Compared as text: a growth rate that is undefined is nan, and nan != nan."""
    assert repr(run_grid(parse([*TINY, "--workers", "1"]))) == repr(rows)
    other_seed = run_grid(parse([*TINY, "--networks", "E", "--replicates", "1", "--seed", "7"]))
    assert repr(other_seed) != repr(rows[:7])


def test_zero_dispersion_cells_reproduce_m1_and_the_protocols_hit_their_targets(rows):
    m1 = {(row["network"], row["replicate"]): row for row in rows if row["model"] == "M1"}
    for row in rows:
        reference = m1[row["network"], row["replicate"]]
        if row["cv"] == 0.0:
            for name in ("c", "r0", "r_analytic", "peak_prevalence", "attack_rate", "peak_time"):
                assert row[name] == pytest.approx(reference[name], rel=1e-9)
        elif row["protocol"] == "i":
            assert row["mean_rate"] == pytest.approx(reference["mean_rate"], rel=1e-12)
            assert row["r0"] < reference["r0"]
        elif row["protocol"] == "ii":
            assert row["r0"] == pytest.approx(reference["r0"], rel=1e-9)
            assert row["r_analytic"] > reference["r_analytic"]
        elif row["protocol"] == "iii":
            assert row["r_analytic"] == pytest.approx(reference["r_analytic"], abs=1e-9)
            assert row["r0"] < reference["r0"]


def test_summary_pairs_only_major_outbreaks_and_is_seeded(rows):
    summary = summarise(rows, seed=113)
    assert repr(summarise(rows, seed=113)) == repr(summary)
    for entry in summary:
        assert entry["pairs"] <= entry["replicates"] == 3
        if entry["cv"] == 0.0 and entry["pairs"]:
            assert entry["difference"] == pytest.approx(0.0, abs=1e-9)
        if entry["pairs"] == 0:
            assert math.isnan(entry["difference"])


def test_run_writes_csvs_that_read_back_and_the_figure(tmp_path, capsys):
    run([*TINY, "--networks", "E", "--replicates", "2", "--workers", "2", "--out", str(tmp_path)])
    runs = read_csv(tmp_path / "runs.csv")
    assert len(runs) == 2 * 7
    assert {"c", "tau_max", "r0", "r0_newman", "r_analytic", "growth_rate"} <= set(runs[0])
    assert runs[0]["major"] in (True, False)
    assert runs[0]["replicate"] == 0
    assert isinstance(runs[0]["attack_rate"], float)
    assert {row["metric"] for row in read_csv(tmp_path / "summary.csv")} >= {"peak_time", "r0"}
    assert (tmp_path / "rq1.png").stat().st_size > 10_000
    assert "14 runs" in capsys.readouterr().out


def test_figure_handles_cells_without_pairs(tmp_path, rows):
    summary = summarise([row | {"major": False} for row in rows], seed=113)
    rq1_figure(summary, tmp_path / "empty.png")
    assert (tmp_path / "empty.png").exists()
