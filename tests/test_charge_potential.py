from src.potential.charge_potential import run_charge_potential_demo


def test_charge_potential_result_schema():
    result = run_charge_potential_demo(N=16, ranks=[1, 2, 4])

    assert result["N"] == 16
    assert result["dx"] == result["L"] / result["N"]
    assert result["baseline_error"] >= 0.0
    assert result["ref_error"] >= 0.0
    assert len(result["rows"]) == 3


def test_charge_potential_rows_have_expected_ranks():
    ranks = [1, 2, 4]
    result = run_charge_potential_demo(N=16, ranks=ranks)

    observed_ranks = [row["rank"] for row in result["rows"]]
    assert observed_ranks == ranks


def test_charge_potential_compression_increases_with_rank():
    result = run_charge_potential_demo(N=16, ranks=[1, 2, 4])

    compressions = [row["compression"] for row in result["rows"]]
    assert compressions[0] < compressions[1] < compressions[2]
