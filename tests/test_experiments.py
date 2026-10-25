import math

import pytest

from run_experiments import (
    CELL,
    difference_figure,
    parse,
    read_csv,
    replicate,
    run,
    run_grid,
    summarise,
)

TINY = ["--networks", "EB", "--n", "1000", "--replicates", "3", "--cv", "0", "1.5"]
RQ2 = ["--n", "600", "--cv", "--theta", "0", "1", "--shuffle", "--alpha", "0", "1"]
OUTCOMES = (
    "c",
    "r0",
    "r_analytic",
    "attack_rate_mp",
    "peak_prevalence",
    "peak_time",
    "attack_rate",
    "growth_rate",
    "deaths",
    "duration",
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rows():
    return run_grid(parse([*TINY, "--workers", "2"]))


@pytest.fixture(scope="module")
def rq2_rows():
    """One replicate in-process: theta x shuffle x alpha on E and B, BBV and its shuffle on W."""
    theta_alpha = parse([*RQ2, "--protocols", "ii", "iii"])
    bbv = parse(["--n", "600", "--networks", "W", "--shuffle", "--protocols", "ii"])
    return [
        *replicate(("E", 0, theta_alpha)),
        *replicate(("B", 0, theta_alpha)),
        *replicate(("W", 0, bbv)),
    ]


def m1_of(rows):
    return {(row["network"], row["replicate"]): row for row in rows if row["model"] == "M1"}


def uniform_rates(cell):
    """alpha = 0 or theta = 0: M2's rates are all equal, so M2 is M1."""
    return cell["alpha"] == 0 or (cell["weights"] == "degree" and cell["level"] == 0)


def outcomes(row):
    """As text: a growth rate that is undefined is nan, and nan != nan."""
    return repr([row[name] for name in OUTCOMES])


def test_grid_has_one_m1_row_per_replicate_and_one_m2_row_per_cell(rows):
    assert len(rows) == 2 * 3 * (1 + 2 * 3)
    assert [row["model"] for row in rows[:7]] == ["M1"] + ["M2"] * 6
    assert {(row["weights"], row["level"], row["protocol"]) for row in rows[1:7]} == {
        ("iid", cv, protocol) for cv in (0.0, 1.5) for protocol in ("i", "ii", "iii")
    }


def test_the_grid_is_reproducible_from_its_seed_whatever_the_workers(rows):
    """Each replicate's streams depend on (seed, replicate) only, so E's first two can run alone."""
    alone = run_grid(parse([*TINY, "--networks", "E", "--replicates", "2", "--workers", "1"]))
    assert repr(alone) == repr(rows[:14])
    assert repr(replicate(("E", 0, parse([*TINY, "--seed", "7"])))) != repr(rows[:7])


def test_zero_dispersion_cells_reproduce_m1_exactly_and_the_protocols_hit_their_targets(rows):
    m1 = m1_of(rows)
    for row in rows:
        reference = m1[row["network"], row["replicate"]]
        if row["level"] == 0.0:
            assert outcomes(row) == outcomes(reference)
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
        if entry["level"] == 0.0 and entry["pairs"]:
            assert entry["difference"] == 0.0
        if entry["pairs"] == 0:
            assert math.isnan(entry["difference"])


def test_run_writes_csvs_that_read_back_and_the_figure(tmp_path, capsys):
    run([*TINY, "--networks", "E", "--replicates", "2", "--workers", "2", "--out", str(tmp_path)])
    runs = read_csv(tmp_path / "runs.csv")
    assert len(runs) == 2 * 7
    assert {*CELL, "c", "tau_max", "r0", "r0_newman", "spearman", "growth_rate"} <= set(runs[0])
    assert runs[0]["major"] in (True, False)
    assert runs[0]["replicate"] == 0
    assert isinstance(runs[0]["attack_rate"], float)
    assert {row["metric"] for row in read_csv(tmp_path / "summary.csv")} >= {"peak_time", "r0"}
    assert (tmp_path / "differences.png").stat().st_size > 10_000
    assert (tmp_path / "differences.pdf").exists()
    assert "14 runs" in capsys.readouterr().out


def test_figure_handles_cells_without_pairs(tmp_path, rows):
    summary = summarise([row | {"major": False} for row in rows], seed=113)
    difference_figure(summary, tmp_path / "empty.png")
    assert (tmp_path / "empty.png").exists()


def test_theta_and_alpha_zero_cells_reproduce_m1_exactly(rq2_rows):
    m1 = m1_of(rq2_rows)
    degenerate = [row for row in rq2_rows if row["model"] == "M2" and uniform_rates(row)]
    assert len(degenerate) == 2 * (4 * 2 + 2 * 2)  # per network: alpha 0, and theta 0 at alpha 1
    for row in degenerate:
        assert row["c"] == 1.0
        assert outcomes(row) == outcomes(m1[row["network"], row["replicate"]])
    zero = [entry for entry in summarise(rq2_rows, seed=113) if uniform_rates(entry)]
    assert all(entry["difference"] == 0.0 for entry in zero if entry["pairs"])


def test_shuffle_nulls_keep_cv_w_exactly_and_lose_the_weight_degree_correlation(rq2_rows):
    twin = {
        tuple(row[name] for name in ("replicate", *CELL) if name != "shuffled"): row
        for row in rq2_rows
        if row["model"] == "M2" and not row["shuffled"]
    }
    shuffled = [row for row in rq2_rows if row["shuffled"]]
    assert len(shuffled) == 2 * 2 * 2 * 2 + 1  # (E, B) x theta x alpha x protocol, and W
    for row in shuffled:
        original = twin[tuple(row[name] for name in ("replicate", *CELL) if name != "shuffled")]
        assert row["cv_w"] == original["cv_w"]
        if row["cv_w"] > 0:
            assert original["spearman"] > 0.3
            assert abs(row["spearman"]) < 0.1


def test_equal_r0_keeps_the_final_size_only_without_weight_degree_correlation(rq2_rows):
    """Message passing under (ii) on B: theta = 1 lowers M2's final size, its shuffle null not."""
    m1 = m1_of(rq2_rows)["B", 0]["attack_rate_mp"]
    final_size = {
        row["shuffled"]: row["attack_rate_mp"]
        for row in rq2_rows
        if (row["network"], row["level"], row["alpha"], row["protocol"]) == ("B", 1.0, 1.0, "ii")
    }
    assert final_size[False] < m1 - 0.1
    assert final_size[True] == pytest.approx(m1, abs=0.03)


def test_bbv_cells_use_the_grown_weights(rq2_rows):
    bbv = [row for row in rq2_rows if row["network"] == "W"]
    assert [(row["model"], row["weights"], row["shuffled"]) for row in bbv] == [
        ("M1", "-", False),
        ("M2", "bbv", False),
        ("M2", "bbv", True),
    ]
    assert bbv[1]["level"] == 1.0 and bbv[1]["cv_w"] > 0.5


def test_difference_figure_draws_theta_and_alpha_sweeps(tmp_path, rq2_rows):
    summary = summarise([row for row in rq2_rows if row["network"] in "EB"], seed=113)
    difference_figure([entry for entry in summary if entry["alpha"] == 1], tmp_path / "theta.png")
    difference_figure([entry for entry in summary if entry["level"] == 1], tmp_path / "alpha.png")
    for name in ("theta.png", "theta.pdf", "alpha.png", "alpha.pdf"):
        assert (tmp_path / name).stat().st_size > 10_000
