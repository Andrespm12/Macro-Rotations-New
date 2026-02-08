"""Backtest, alpha factor, cross-asset, and portfolio optimization charts."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict
from analytics.quant import calculate_strategy_performance
from core.logger import get_logger

logger = get_logger(__name__)

def run_backtest_plot(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Runs a vectorised backtest for multiple strategies and plots the results."""
    logger.info("Running Backtest & Projections...")

    metrics, curves = calculate_strategy_performance(df, prices)
    if curves is None or curves.empty:
        logger.warning("Backtest failed: Missing assets.")
        return None

    # Simplified Projection: Monte Carlo using Full History
    projections = pd.DataFrame()
    proj_metrics = {}

    future_days = 252
    n_sims = 1000
    last_date = curves.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='B')
    np.random.seed(42)

    for strat in curves.columns:
        series = curves[strat]
        rets = series.pct_change().infer_objects(copy=False).dropna()
        mu = rets.mean()
        sigma = rets.std()
        last_price = series.iloc[-1]

        ret_sim = np.random.normal(mu, sigma, (future_days, n_sims))
        price_paths = last_price * (1 + ret_sim).cumprod(axis=0)
        median_path = np.median(price_paths, axis=1)

        full_proj = np.concatenate(([last_price], median_path))
        projections[strat] = full_proj

        final_val = median_path[-1]
        exp_ret = (final_val / last_price) - 1
        proj_metrics[strat] = exp_ret

    # Plotting
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0])
    colors = {
        "Core-Satellite (CCI)": "#1f77b4",
        "Macro Regime": "#2ca02c",
        "Liquidity Valve": "#d62728",
        "Breadth Trend": "#9467bd",
        "Consumer Rotation": "#ff7f0e",
        "VIX Filter": "#e377c2",
        "Sector Leaders (Top 3)": "#8c564b",
        "Vol Control (12%)": "#17becf",
        "SPY (Hold)": "black"
    }

    for strat in curves.columns:
        cagr = metrics[strat]["CAGR"]
        label = f"{strat} (CAGR: {cagr:.1%})"
        color = colors.get(strat, "grey")
        style = "--" if "Hold" in strat else "-"
        width = 3 if "Core-Satellite" in strat else (1.5 if "Hold" in strat else 1.5)
        alpha = 1.0 if "Core-Satellite" in strat else 0.7

        ax0.plot(curves.index, curves[strat], label=label, color=color, linestyle=style, linewidth=width, alpha=alpha)

        if strat in projections.columns:
            ax0.plot(future_dates, projections[strat], color=color, linestyle=":", linewidth=width, alpha=0.6)
            ax0.scatter(future_dates[-1], projections[strat].iloc[-1], color=color, s=20)

    ax0.set_title("Backtest & 1-Year Projection (Median Path)", fontsize=16, weight='bold', color='black')
    ax0.set_yscale('log')
    ax0.legend(loc="upper left", fontsize=10, frameon=True, facecolor='white', edgecolor='grey')
    ax0.grid(True, which="both", alpha=0.3, color='grey', linestyle=':')
    ax0.set_ylabel("Cumulative Return (Log Scale)", fontsize=12)
    ax0.axvline(last_date, color='black', linestyle='-', linewidth=1)
    ax0.text(last_date, ax0.get_ylim()[0], "  TODAY", rotation=90, va='bottom', weight='bold')

    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')

    table_data = [["Strategy", "Hist. CAGR", "Sharpe", "Max Drawdown", "Exp. CAGR (1Y)"]]
    sorted_strats = sorted(metrics.keys(), key=lambda x: metrics[x]['Sharpe'], reverse=True)

    for strat in sorted_strats:
        m = metrics[strat]
        exp = proj_metrics.get(strat, 0)
        table_data.append([
            strat,
            f"{m['CAGR']:.1%}",
            f"{m['Sharpe']:.2f}",
            f"{m['MaxDD']:.1%}",
            f"{exp:.1%}"
        ])

    table = ax1.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.12, 0.12, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    return fig

def plot_alpha_factors_page(prices: pd.DataFrame, macro: pd.DataFrame, alpha_data: Dict) -> plt.Figure:
    """Generates Page 6: Institutional Alpha Factors."""
    logger.info("Generating Alpha Factors Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 16)) # Increased height
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])

    # 1. Net Liquidity vs SPY
    ax1 = fig.add_subplot(gs[0])
    nl_data = alpha_data.get("net_liquidity", {})
    if not nl_data.get("series", pd.Series()).empty:
        nl = nl_data["series"]
        spy = prices["SPY"].reindex(nl.index).ffill()

        # Plot Net Liquidity (Left)
        color_nl = 'darkblue'
        ax1.plot(nl.index, nl, color=color_nl, linewidth=2, label="Net Liquidity ($Trillion)")
        ax1.set_ylabel("Net Liquidity ($T)", color=color_nl, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_nl)

        # Plot SPY (Right)
        ax1_twin = ax1.twinx()
        color_spy = 'black'
        ax1_twin.plot(spy.index, spy, color=color_spy, linestyle='--', alpha=0.6, label="S&P 500 (Right)")
        ax1_twin.set_ylabel("S&P 500 Price", color=color_spy, fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor=color_spy)

        # Status Validation
        status = nl_data.get("status", "N/A")
        raw_val = nl_data.get("latest", 0)
        ax1.set_title(f"1. The Real Liquidity Engine (Fed - TGA - RRP)\nCurrent: ${raw_val:.2f}T | Trend: {status}", fontsize=14, weight='bold')

        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Net Liquidity Data Unavailable", ha='center')

    # 2. Volatility Term Structure (Crash Signal)
    ax2 = fig.add_subplot(gs[1])
    vol_data = alpha_data.get("vol_structure", {})
    if not vol_data.get("ratio", pd.Series()).empty:
        ratio = vol_data["ratio"]
        ax2.plot(ratio.index, ratio, color='purple', label="VIX / VIX3M Ratio")

        # Thresholds
        ax2.axhline(1.0, color='black', linestyle='-', linewidth=1, label="Contango/Backwardation Flip")
        ax2.fill_between(ratio.index, ratio, 1.0, where=(ratio > 1.0), color='red', alpha=0.3, label="Crash Risk (>1.0)")
        ax2.fill_between(ratio.index, ratio, 1.0, where=(ratio <= 1.0), color='green', alpha=0.1, label="Normal (<1.0)")

        curr_sig = vol_data.get("signal", "N/A")
        color_sig = 'red' if "CRASH" in curr_sig else 'green'
        ax2.set_title(f"2. Volatility Term Structure (Crash Signal)\nStatus: {curr_sig}", fontsize=12, weight='bold', color=color_sig)
        ax2.set_ylabel("VIX / VIX3M Ratio")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Volatility Data Missing", ha='center')

    # 3. Tail Risk (SKEW)
    ax3 = fig.add_subplot(gs[2])
    skew_data = alpha_data.get("tail_risk", {})
    if not skew_data.get("series", pd.Series()).empty:
        skew = skew_data["series"]
        ax3.plot(skew.index, skew, color='darkred', label="CBOE SKEW Index")

        # Zones
        ax3.axhline(135, color='red', linestyle='--', label="High Risk (>135)")
        ax3.axhline(115, color='green', linestyle='--', label="Complacency (<115)")

        curr_skew = skew_data.get("signal", "N/A")
        ax3.set_title(f"3. Tail Risk Monitor (Whale Positioning)\nStatus: {curr_skew}", fontsize=12, weight='bold')
        ax3.set_ylabel("SKEW Index")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "SKEW Data Missing", ha='center')

    plt.tight_layout()

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.12)

    interp_text = "Institutional Flows & Risk:\n"

    # 1. Net Liquidity
    nl_lat = alpha_data.get("net_liquidity", {}).get("latest", 0)
    interp_text += f"• Net Liquidity: ${nl_lat:.2f}T. Tracks Fed Balance Sheet - TGA - RRP. Rising = Asset Support.\n"

    # 2. VIX Term Structure
    v_rat = alpha_data.get("vol_structure", {}).get("ratio")
    if not isinstance(v_rat, pd.Series): v_rat = pd.Series([0])
    curr_v = v_rat.iloc[-1] if not v_rat.empty else 0

    if curr_v > 1.0:
        interp_text += f"• Vol Term Structure: BACKWARDATION ({curr_v:.2f}). CRASH WARNING. Immediate fear > future fear.\n"
    else:
        interp_text += f"• Vol Term Structure: CONTANGO ({curr_v:.2f}). Normal market structure.\n"

    # 3. SKEW
    s_val = alpha_data.get("tail_risk", {}).get("series")
    if not isinstance(s_val, pd.Series): s_val = pd.Series([0])
    curr_s = s_val.iloc[-1] if not s_val.empty else 0

    if curr_s > 135: interp_text += f"• Tail Risk (SKEW): HIGH ({curr_s:.0f}). Whales are hedging against a crash.\n"
    elif curr_s < 115: interp_text += f"• Tail Risk (SKEW): COMPLACENT ({curr_s:.0f}). Vulnerable to shocks.\n"
    else: interp_text += f"• Tail Risk (SKEW): NORMAL ({curr_s:.0f}).\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_cross_asset_page(prices: pd.DataFrame, corr_data: Dict) -> plt.Figure:
    """Generates Page 9: Cross-Asset Regime."""
    logger.info("Generating Cross-Asset Correlation Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 16)) # Increased height
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8])

    # 1. Correlation Matrix Heatmap
    ax1 = fig.add_subplot(gs[0])
    matrix = corr_data.get("matrix", pd.DataFrame())

    if not matrix.empty:
        cax = ax1.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)

        # Labels
        labels = matrix.columns
        ax1.set_xticks(np.arange(len(labels)))
        ax1.set_yticks(np.arange(len(labels)))
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_yticklabels(labels, fontsize=10)

        # Annotate
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=12, weight='bold')

        ax1.set_title("1. Cross-Asset Correlation Matrix (30-Day Rolling)\nGreen/Red = Diversification | Dark = High Correlation", fontsize=14, weight='bold')
    else:
        ax1.text(0.5, 0.5, "Insufficient Data for Correlation Matrix", ha='center')

    # 2. SPY vs TLT Rolling Correlation (Regime Check)
    ax2 = fig.add_subplot(gs[1])
    rolling = corr_data.get("spy_tlt_rolling", pd.Series())

    if not rolling.empty:
        ax2.plot(rolling.index, rolling, color='black', linewidth=1.5, label="SPY vs TLT (6M Rolling)")

        # Zones
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.axhline(0.5, color='red', linestyle=':', label="danger (>0.5)")
        ax2.axhline(-0.5, color='green', linestyle=':', label="Diversified (<-0.5)")

        # Fill
        ax2.fill_between(rolling.index, rolling, 0.5, where=(rolling > 0.5), color='red', alpha=0.3, label="Inflation/Rate Risk")
        ax2.fill_between(rolling.index, rolling, -0.5, where=(rolling < -0.5), color='green', alpha=0.2, label="Deflation/Growth Risk")

        curr = rolling.iloc[-1]
        regime = "INFLATION FEAR" if curr > 0.5 else ("DEFLATION/GROWTH FEAR" if curr < -0.5 else "NORMAL DIVERSIFICATION")

        ax2.set_title(f"2. Stock-Bond Correlation Regime\nCurrent: {curr:.2f} -> {regime}", fontsize=12, weight='bold')
        ax2.set_ylabel("Correlation")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "SPY/TLT Data Missing", ha='center')

    plt.tight_layout()

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15)

    interp_text = "Cross-Asset Regime:\n"
    interp_text += "• Correlation Matrix: Dark Red = Assets moving together (Systemic Risk). Blue = Diversification working.\n"

    # SPY vs TLT
    rolling = corr_data.get("spy_tlt_rolling", pd.Series())
    if not rolling.empty:
        curr_c = rolling.iloc[-1]
        interp_text += f"• Stock/Bond Correlation: {curr_c:.2f}. "
        if curr_c > 0.5: interp_text += "HIGHLY CORRELATED. Bonds will NOT protect portfolios. Inflation risk dominant.\n"
        elif curr_c < -0.5: interp_text += "NEGATIVELY CORRELATED. Magnificent diversification. Growth risk dominant.\n"
        else: interp_text += "UNCORRELATED. Standard diversification environment.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_efficient_frontier_page(optimization_data: Dict) -> plt.Figure:
    """Generates Page 10: Portfolio Optimization Lab."""
    logger.info("Generating Efficient Frontier Page...")
    plt.style.use('default')

    if not optimization_data or "results" not in optimization_data:
        return None

    results = optimization_data["results"]
    max_sharpe = optimization_data["max_sharpe"]
    min_vol = optimization_data["min_vol"]
    assets = optimization_data["assets"]

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 0.5])

    # 1. Efficient Frontier Scatter
    ax1 = fig.add_subplot(gs[0])
    sc = ax1.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.5)
    fig.colorbar(sc, ax=ax1, label="Sharpe Ratio")

    # Mark Max Sharpe
    ax1.scatter(max_sharpe["metrics"][1], max_sharpe["metrics"][0], marker='*', color='r', s=500, label=f"Max Sharpe ({max_sharpe['metrics'][2]:.2f})")

    # Mark Min Vol
    ax1.scatter(min_vol["metrics"][1], min_vol["metrics"][0], marker='o', color='b', s=200, label=f"Min Volatility (Vol: {min_vol['metrics'][1]:.1%})", edgecolors='white', linewidth=2)

    ax1.set_title("1. THE EFFICIENT FRONTIER (5000 Simulated Portfolios)\nRisk vs Return Trade-off", fontsize=16, weight='bold')
    ax1.set_xlabel("Annualized Volatility (Risk)")
    ax1.set_ylabel("Annualized Return")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Optimal Allocation Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    table_data = []
    # Header
    table_data.append(["Asset", "Min Volatility", "Max Sharpe (Optimal)"])

    for asset in assets:
        min_w = min_vol["weights"].get(asset, 0)
        max_w = max_sharpe["weights"].get(asset, 0)
        # Fix formatting for zero weights
        min_str = f"{min_w:.1%}" if min_w > 0.001 else "-"
        max_str = f"{max_w:.1%}" if max_w > 0.001 else "-"
        table_data.append([asset, min_str, max_str])

    table = ax2.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.0)

    # Style Header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040')

    ax2.set_title("2. OPTIMAL ASSET ALLOCATION (Mean-Variance)", fontsize=14, weight='bold')

    plt.tight_layout()

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15)

    interp_text = "Portfolio Optimization Insights:\n"
    # Max Sharpe Asset
    if "weights" in max_sharpe:
         best_asset = max(max_sharpe["weights"], key=max_sharpe["weights"].get)
         best_w = max_sharpe["weights"][best_asset]
         interp_text += f"• Max Sharpe Portfolio: Heaviest allocation is {best_asset} ({best_w:.1%}). Maximizes risk-adjusted return.\n"

    # Min Vol Asset
    if "weights" in min_vol:
         safe_asset = max(min_vol["weights"], key=min_vol["weights"].get)
         safe_w = min_vol["weights"][safe_asset]
         interp_text += f"• Min Volatility Portfolio: Heaviest allocation is {safe_asset} ({safe_w:.1%}). Focuses on capital preservation.\n"

    interp_text += "• Efficient Frontier: Portfolios on the top-left edge offer the highest return for a given level of risk."

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig
