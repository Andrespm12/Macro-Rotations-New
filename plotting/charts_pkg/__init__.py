"""
Chart subpackage - re-exports all chart functions for backward compatibility.

Usage:
    from plotting.charts_pkg import plot_risk_macro_dashboard
    # or import the full set
    from plotting.charts_pkg import *
"""
from plotting.charts_pkg.macro import (
    plot_risk_macro_dashboard,
    plot_macro_radar_chart,
    plot_monetary_plumbing,
    plot_forward_models,
    plot_predictive_models_page,
)
from plotting.charts_pkg.quant import (
    plot_quant_lab_dashboard,
    plot_monte_carlo_cone,
    plot_stochastic_page,
    plot_mean_reversion_page,
    plot_microstructure_page,
    plot_antifragility_page,
    plot_scenario_page,
)
from plotting.charts_pkg.portfolio import (
    run_backtest_plot,
    plot_alpha_factors_page,
    plot_cross_asset_page,
    plot_efficient_frontier_page,
)
from plotting.charts_pkg.global_macro import (
    plot_valuation_page,
    plot_inflation_swap_curve,
    plot_global_macro_fx_page,
)

__all__ = [
    "run_backtest_plot",
    "plot_risk_macro_dashboard",
    "plot_macro_radar_chart",
    "plot_monetary_plumbing",
    "plot_forward_models",
    "plot_quant_lab_dashboard",
    "plot_alpha_factors_page",
    "plot_cross_asset_page",
    "plot_efficient_frontier_page",
    "plot_predictive_models_page",
    "plot_monte_carlo_cone",
    "plot_stochastic_page",
    "plot_mean_reversion_page",
    "plot_microstructure_page",
    "plot_antifragility_page",
    "plot_scenario_page",
    "plot_valuation_page",
    "plot_inflation_swap_curve",
    "plot_global_macro_fx_page",
]
