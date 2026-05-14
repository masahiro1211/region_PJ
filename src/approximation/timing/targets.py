"""時間計測対象 callable の定義。"""

from __future__ import annotations

from src.approximation.cp_coulomb import (
    apply_cp_rho_E,
    apply_cp_rho_E_rpca,
    apply_cp_rho_V,
    apply_cp_rho_V_rpca,
)
from src.approximation.timing.types import (
    TimingBenchmarkInputs,
    TimingTarget,
)
from src.approximation.torch_kernels import (
    apply_exp_sum_3d_full,
    apply_exp_sum_3d_lowrank,
    apply_exp_sum_3d_lowrank_naive,
    apply_exp_sum_3d_rpca,
    apply_exp_sum_3d_rpca_l_only,
    apply_exp_sum_3d_rpca_s_only,
)


def iter_timing_targets(
    inputs: TimingBenchmarkInputs,
) -> list[TimingTarget]:
    """10 手法の時間計測対象 callable を返す。

    Parameters
    ----------
    inputs : TimingBenchmarkInputs
        ``prepare_timing_benchmark_inputs`` が返す PyTorch 入力一式。

    Returns
    -------
    targets : list[TimingTarget]
        保存順に並んだ時間計測対象。
    """
    return [
        TimingTarget(
            "full",
            lambda: apply_exp_sum_3d_full(
                inputs.rho_pt,
                inputs.full_kernels_pt,
            ),
        ),
        TimingTarget(
            "naive",
            lambda: apply_exp_sum_3d_lowrank_naive(
                inputs.rho_pt,
                inputs.lowrank_data_pt,
            ),
        ),
        TimingTarget(
            "lowrank",
            lambda: apply_exp_sum_3d_lowrank(
                inputs.rho_pt,
                inputs.lowrank_data_pt,
            ),
        ),
        TimingTarget(
            "rpca",
            lambda: apply_exp_sum_3d_rpca(
                inputs.rho_pt,
                inputs.rpca_lowrank_only_data_pt,
                inputs.rpca_dense_data_pt,
            ),
        ),
        TimingTarget(
            "rpca_l",
            lambda: apply_exp_sum_3d_rpca_l_only(
                inputs.rho_pt,
                inputs.rpca_lowrank_only_data_pt,
                inputs.rpca_dense_data_pt,
            ),
        ),
        TimingTarget(
            "rpca_s",
            lambda: apply_exp_sum_3d_rpca_s_only(
                inputs.rho_pt,
                inputs.rpca_dense_data_pt,
            ),
        ),
        TimingTarget(
            "cp_v",
            lambda: apply_cp_rho_V(
                inputs.rho_terms_pt,
                inputs.kernel_list_pt,
            ),
        ),
        TimingTarget(
            "cp_e",
            lambda: apply_cp_rho_E(
                inputs.rho_terms_pt,
                inputs.kernel_list_pt,
                inputs.dx,
            ),
        ),
        TimingTarget(
            "cp_rpca_v",
            lambda: apply_cp_rho_V_rpca(
                inputs.rho_terms_pt,
                inputs.rpca_lowrank_only_data_pt,
                inputs.rpca_dense_data_pt,
            ),
        ),
        TimingTarget(
            "cp_rpca_e",
            lambda: apply_cp_rho_E_rpca(
                inputs.rho_terms_pt,
                inputs.rpca_lowrank_only_data_pt,
                inputs.rpca_dense_data_pt,
                inputs.dx,
            ),
        ),
    ]
