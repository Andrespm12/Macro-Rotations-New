"""Global macro, FX, valuation, and inflation charts."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict

from core.logger import get_logger

logger = get_logger(__name__)


def plot_valuation_page(df: pd.DataFrame, fundamentals: Dict) -> plt.Figure:
    """Generates Page: Valuation & Real Rates (Real Yields & ERP)."""
    logger.info("Generating Valuation & Real Rates Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    # 1. Real Interest Rates (10Y - Breakeven)
    ax1 = fig.add_subplot(gs[0])
    if "CF_Real_Yield" in df.columns:
        real_yield = df["CF_Real_Yield"].dropna()
        ax1.plot(real_yield.index, real_yield, color='blue', linewidth=2, label="US 10Y Real Yield")

        # Zones
        ax1.axhline(2.0, color='red', linestyle='--', label="Restrictive (>2.0%)")
        ax1.axhline(0.5, color='green', linestyle='--', label="Accommodative (<0.5%)")
        ax1.axhline(0.0, color='black', linewidth=1)

        # Fill
        ax1.fill_between(real_yield.index, real_yield, 2.0, where=(real_yield > 2.0), color='red', alpha=0.2)
        ax1.fill_between(real_yield.index, real_yield, 0.0, where=(real_yield < 0.0), color='green', alpha=0.1, label="Negative Real Rates")

        curr_real = real_yield.iloc[-1]
        status = "RESTRICTIVE" if curr_real > 2.0 else ("NEUTRAL" if curr_real > 0.5 else "STIMULATIVE")

        ax1.set_title(f"1. Real Interest Rates (The Cost of Capital)\nCurrent: {curr_real:.2f}% -> {status}", fontsize=14, weight='bold')
        ax1.set_ylabel("Real Yield (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Commentary Box
        text = (
            "Implications:\n"
            "• High Real Rates (>2%): Tight financial conditions. Headwind for Gold, Tech, & Multiples.\n"
            "• Low/Negative Real Rates: Loose conditions. Tailwind for Hard Assets & Speculation.\n"
            "• Trend is key: Rapidly rising real rates often trigger deleveraging events."
        )
        ax1.text(0.02, 0.05, text, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey', boxstyle='round,pad=0.5'))

    else:
        ax1.text(0.5, 0.5, "Real Yield Data Missing", ha='center')

    # 2. Equity Risk Premium (ERP) - The Fed Model
    ax2 = fig.add_subplot(gs[1])

    pe = fundamentals.get("SPY_PE")
    # Fetch latest 10Y Yield from DF if not in fundamentals (it should be in DF)
    nominal_10y = df["10Y_Yield"].iloc[-1] if "10Y_Yield" in df.columns else None

    if pe and nominal_10y:
        earnings_yield = (1 / pe) * 100
        erp = earnings_yield - nominal_10y

        # Bar Chart
        labels = ['10Y Treasury Yield', 'S&P 500 Earnings Yield (1/PE)']
        values = [nominal_10y, earnings_yield]
        colors = ['red', 'green']

        bars = ax2.barh(labels, values, color=colors, alpha=0.7)
        ax2.set_xlim(0, max(values) * 1.3)

        # Annotate Values
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", va='center', fontsize=12, weight='bold')

        # Draw Spread (ERP)
        # We can visualize this as a bracket or text
        ax2.text(max(values) * 0.5, 0.5, f"Equity Risk Premium (Spread): {erp:.2f}%",
                 ha='center', va='center', transform=ax2.transAxes, fontsize=16, weight='bold',
                 bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=1'))

        # Status
        valuation = "ATTRACTIVE (Stocks Cheap)" if erp > 3.0 else ("FAIR VALUE" if erp > 0 else "EXPENSIVE (Stocks Rich)")
        color_val = 'green' if erp > 0 else 'red'

        ax2.set_title(f"2. Equity Risk Premium (Fed Model Snapshot)\nValuation: {valuation}", fontsize=14, weight='bold', color=color_val)
        ax2.set_xlabel("Yield (%)")

        # ERP Context
        context_text = (
            "Interpretation:\n"
            "• ERP = Earnings Yield - Risk Free Rate.\n"
            "• Positive ERP: Stocks offer excess return over bonds.\n"
            "• Negative ERP: Stocks yield less than bonds (Speculative territory needs Growth)."
        )
        ax2.text(0.7, 0.1, context_text, transform=ax2.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))

    else:
        ax2.text(0.5, 0.5, "Insufficient Data for Valuation (PE or Yield Missing)", ha='center')

    plt.tight_layout()
    return fig


def plot_inflation_swap_curve(df: pd.DataFrame) -> plt.Figure:
    """Generates Page: Inflation Expectations Term Structure (Swaps Proxy)."""
    logger.info("Generating Inflation Expectations Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    # 1. Term Structure: 5Y vs 10Y vs 5Y5Y
    ax1 = fig.add_subplot(gs[0])

    # Check Data Availability
    has_5y = "5Y_Breakeven" in df.columns
    has_10y = "10Y_Breakeven" in df.columns
    has_5y5y = "5Y5Y_Forward" in df.columns

    if has_5y5y or has_10y:
        if has_5y:
            breakeven_5 = df["5Y_Breakeven"].dropna()
            ax1.plot(breakeven_5.index, breakeven_5, color='green', alpha=0.6, label="5-Year Breakeven (Near Term)")

        if has_10y:
            breakeven_10 = df["10Y_Breakeven"].dropna()
            ax1.plot(breakeven_10.index, breakeven_10, color='blue', alpha=0.6, label="10-Year Breakeven (Medium Term)")

        if has_5y5y:
            fwd_5y5y = df["5Y5Y_Forward"].dropna()
            ax1.plot(fwd_5y5y.index, fwd_5y5y, color='red', linewidth=2, label="5Y, 5Y Forward (Long Term Anchor)")

            # Fill between 5Y and 5Y5Y to show curve slope if both exist
            if has_5y:
                 common = fwd_5y5y.index.intersection(breakeven_5.index)
                 ax1.fill_between(common, fwd_5y5y.loc[common], breakeven_5.loc[common], color='gray', alpha=0.1, label="Term Premium / Curve Slope")

        ax1.axhline(2.0, color='black', linestyle='--', linewidth=1.5, label="Fed Target (2.0%)")

        # Get latest values for title
        last_val = "N/A"
        status = "ANCHORED"
        if has_5y5y:
            curr = fwd_5y5y.iloc[-1]
            last_val = f"{curr:.2f}%"
            if curr > 2.5: status = "DE-ANCHORING (High)"
            elif curr < 1.5: status = "DEFLATIONARY RISK"

        ax1.set_title(f"1. Inflation Expectations Term Structure\nLong-Term Anchor (5Y5Y): {last_val} -> {status}", fontsize=14, weight='bold')
        ax1.set_ylabel("Inflation Rate (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # --- Interpretation Box ---
        interp_text = "Interpretation:\n"

        # 1. Slope (5Y vs 5Y5Y)
        if has_5y5y and has_5y:
             slope = fwd_5y5y.iloc[-1] - breakeven_5.iloc[-1]
             if slope > 0.1:
                 interp_text += "• Curve Slope: CONTANGO (Normal). Market sees inflation rising to long-term avg.\n"
             elif slope < -0.1:
                 interp_text += "• Curve Slope: INVERTED (Front-Loaded). High short-term inflation expected to cool.\n"
             else:
                 interp_text += "• Curve Slope: FLAT. Inflation expectations are uniform across horizons.\n"

        # 2. Anchor Level
        if has_5y5y:
            curr_anchor = fwd_5y5y.iloc[-1]
            if curr_anchor > 2.5:
                interp_text += "• Anchor Status: ELEVATED. Long-term expectations > 2.5% (Fed concern).\n"
            elif curr_anchor < 1.8:
                interp_text += "• Anchor Status: LOW. Risk of deflationary trap.\n"
            else:
                 interp_text += "• Anchor Status: STABLE. Near Fed target (2.0%).\n"

        ax1.text(0.02, 0.1, interp_text, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='grey'))
    else:
        ax1.text(0.5, 0.5, "Inflation Swap Data Missing (Check Tickers)", ha='center')

    # 2. Inflation Risk Premium (5Y5Y vs Spot CPI)
    # Allows us to see if the market is pricing higher inflation than current realized
    ax2 = fig.add_subplot(gs[1])

    if has_5y5y and "CPI_YoY" in df.columns:
        fwd = df["5Y5Y_Forward"].dropna()
        cpi = df["CPI_YoY"].dropna() * 100 # Scale to %

        # Align
        common_idx = fwd.index.intersection(cpi.index)
        fwd = fwd.loc[common_idx]
        cpi = cpi.loc[common_idx]

        spread = fwd - cpi

        ax2.plot(spread.index, spread, color='purple', label="Inflation Risk Premium (5Y5Y Fwd - Current CPI)")
        ax2.axhline(0, color='black', linewidth=1)

        ax2.fill_between(spread.index, spread, 0, where=(spread > 0), color='green', alpha=0.2, label="Market Expects HIGHER Inflation")
        ax2.fill_between(spread.index, spread, 0, where=(spread < 0), color='red', alpha=0.2, label="Market Expects LOWER Inflation")

        curr_spread = spread.iloc[-1]

        ax2.set_title(f"2. Inflation Term Premium (Expectations vs Reality)\nSpread: {curr_spread:.2f}%", fontsize=14, weight='bold')
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Text Context
        text_ctx = (
            "Interpretation:\n"
            "• Positive Spread: Market believes current inflation is temporary/low and will rise to long-term avg.\n"
            "• Negative Spread: Market believes current inflation is too high and will fall (Mean Reversion).\n"
            "• Deep Negative: High conviction in Disinflation."
        )
        ax2.text(0.02, 0.1, text_ctx, transform=ax2.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))

    else:
         ax2.text(0.5, 0.5, "Data Missing for Premium Calculation", ha='center')

    plt.tight_layout()
    return fig


def plot_global_macro_fx_page(df: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame, global_flows: Dict) -> plt.Figure:
    """Generates Page: Global Macro, FX & Capital Flows."""
    logger.info("Generating Global FX & Rates Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # 1. Global Sovereign Yields Overlay (10Y)
    ax1 = fig.add_subplot(gs[0])

    # US 10Y
    if "10Y_Yield" in macro.columns:
        us10 = macro["10Y_Yield"].dropna()
        ax1.plot(us10.index, us10, color='blue', linewidth=2, label="US 10Y Treasury")

    # Germany 10Y
    if "Germany_10Y" in macro.columns:
        de10 = macro["Germany_10Y"].dropna()
        ax1.plot(de10.index, de10, color='black', linestyle='--', label="Germany 10Y Bund")

    # Japan 10Y
    if "Japan_10Y" in macro.columns:
        jp10 = macro["Japan_10Y"].dropna()
        ax1.plot(jp10.index, jp10, color='red', linestyle='--', label="Japan 10Y JGB")

    # UK 10Y
    if "UK_10Y" in macro.columns:
        uk10 = macro["UK_10Y"].dropna()
        ax1.plot(uk10.index, uk10, color='green', linestyle=':', label="UK 10Y Gilt")

    ax1.set_title("1. Global Sovereign Bond Yields (10Y Nominal)\nMonitor: Yield Divergence drives Capital Flows", fontsize=14, weight='bold')
    ax1.set_ylabel("Yield (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. The Carry Trade (US-JP Spread vs USD/JPY)
    ax2 = fig.add_subplot(gs[1])

    if "10Y_Yield" in macro.columns and "Japan_10Y" in macro.columns and "JPY=X" in prices.columns:
        # Align Data
        us = macro["10Y_Yield"]
        jp = macro["Japan_10Y"]
        fx = prices["JPY=X"]

        common = us.index.intersection(jp.index).intersection(fx.index)

        spread = (us.loc[common] - jp.loc[common])
        usd_jpy = fx.loc[common]

        # Plot Spread (Left Axis)
        color_spread = 'darkblue'
        ax2.plot(spread.index, spread, color=color_spread, label="US-Japan 10Y Yield Spread")
        ax2.set_ylabel("Yield Spread (%)", color=color_spread, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_spread)

        # Plot FX (Right Axis)
        ax2_twin = ax2.twinx()
        color_fx = 'darkred'
        ax2_twin.plot(usd_jpy.index, usd_jpy, color=color_fx, linestyle='--', alpha=0.7, label="USD/JPY Exchange Rate")
        ax2_twin.set_ylabel("USD/JPY", color=color_fx, fontsize=12)
        ax2_twin.tick_params(axis='y', labelcolor=color_fx)

        # Status
        carry_status = global_flows.get("japan_carry", {}).get("status", "N/A")

        ax2.set_title(f"2. The 'Carry Trade' Engine (Yield Spread vs FX)\nStatus: {carry_status}", fontsize=14, weight='bold')

        # Legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient Data for Carry Trade Analysis", ha='center')

    # 3. Currency Relative Performance (1-Year Normalized)
    ax3 = fig.add_subplot(gs[2])

    currencies = {
        "USD (DXY)": "DX-Y.NYB" if "DX-Y.NYB" in prices.columns else "UUP",
        "Euro (FXE)": "FXE",
        "Yen (FXY)": "FXY",
        "Yuan (CNY=X)": "CNY=X"
    }

    # Filter for existing
    valid_curr = {k: v for k, v in currencies.items() if v in prices.columns}

    if valid_curr:
        # Get last 252 days
        start_idx = -252

        for name, ticker in valid_curr.items():
            series = prices[ticker].iloc[start_idx:].dropna()
            if not series.empty:
                # Normalize to 100
                norm = (series / series.iloc[0]) * 100

                # Invert logic for standard pair convention if needed?
                # DXY: Long USD.
                # FXE: Long Euro (Short USD).
                # FXY: Long Yen (Short USD).
                # CNY=X: USD/CNY usually. If it's USD/CNY, rising = Weak Yuan.
                # Let's plot raw ETF performance for simplicity as they represent "Long that Currency against USD" (except DXY).
                # Note: CNY=X in Yahoo is usually USD/CNY.

                linewidth = 2 if "USD" in name else 1.5
                alpha = 1.0 if "USD" in name else 0.7

                ax3.plot(norm.index, norm, label=name, linewidth=linewidth, alpha=alpha)

        ax3.axhline(100, color='black', linestyle='--', linewidth=1)
        ax3.set_title("3. Currency Momentum (1-Year Relative Performance, Normalized=100)", fontsize=14, weight='bold')
        ax3.set_ylabel("Rel Perf")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

    else:
        ax3.text(0.5, 0.5, "Currency Data Missing", ha='center')

    plt.tight_layout()

    # Interpretation
    fig.subplots_adjust(bottom=0.12)
    interp_text = "Global Macro & FX Insights:\n"

    # 1. Yield Divergence
    if "10Y_Yield" in macro.columns:
        us_y = macro["10Y_Yield"].dropna().iloc[-1]
        interp_text += f"• US Yields: {us_y:.2f}%. "
        if "Germany_10Y" in macro.columns:
             de_y = macro["Germany_10Y"].dropna().iloc[-1]
             diff = us_y - de_y
             interp_text += f"vs Bunds: {diff:.2f}% spread. "

    # 2. Carry
    carry_stat = global_flows.get("japan_carry", {}).get("status", "N/A")
    if "UNWINDING" in carry_stat:
        interp_text += "\n• CARRY UNWIND ALERT: JPY Strengthening while Yield Spread compresses. Risk-Off signal."
    elif "ACCELERATING" in carry_stat:
        interp_text += "\n• CARRY TRADE ON: JPY Weakening + Yield Spread widening. Supports Global Liquidity."

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))

    return fig
