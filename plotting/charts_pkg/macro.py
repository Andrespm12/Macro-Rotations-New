"""Macro regime, risk, and forward-looking model charts."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm
from typing import Dict

from analytics.rotations import calculate_rrg_metrics, predict_sector_rotation, calculate_seasonality
from analytics.macro_models import calculate_macro_radar, calculate_regime_gmm
from analytics.quant import calculate_systemic_risk_pca
from core.logger import get_logger

logger = get_logger(__name__)


def plot_risk_macro_dashboard(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates Page 3: Macro Risk & Sector Rotation."""
    logger.info("Generating Risk & Macro Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.5])

    # 1. Yield Curve
    ax1 = fig.add_subplot(gs[0])
    has_yields = "10Y_Yield" in df.columns and "2Y_Yield" in df.columns
    if has_yields:
        yc = (df["10Y_Yield"] - df["2Y_Yield"]).dropna()
        if not yc.empty:
            ax1.plot(yc.index, yc, color='black', label="10Y-2Y Spread")
            ax1.axhline(0, color='red', linestyle='--', linewidth=1)
            ax1.fill_between(yc.index, yc, 0, where=(yc < 0), color='red', alpha=0.3)
            ax1.fill_between(yc.index, yc, 0, where=(yc > 0), color='green', alpha=0.1)
            ax1.set_title("1. Yield Curve (10Y - 2Y): Recession Watch", fontsize=12, weight='bold')
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Yield Curve Data Missing", ha='center')

    # 2. Credit Stress
    ax2 = fig.add_subplot(gs[1])
    if "HY_Spread" in df.columns:
        hy = df["HY_Spread"].dropna()
        ax2.plot(hy.index, hy, color='purple', label="High Yield Option-Adjusted Spread")
        ax2.axhline(hy.mean(), color='orange', linestyle='--', label="Avg Spread")
        ax2.set_title("2. Credit Stress (High Yield Spreads)", fontsize=12, weight='bold')
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Credit Spread Data Missing", ha='center')

    # 3. Bond Market Fear
    ax3 = fig.add_subplot(gs[2])
    if "MOVE_Index" in df.columns:
        move = df["MOVE_Index"].dropna()
        ax3.plot(move.index, move, color='blue', label="MOVE Index (Bond Volatility)")
        ax3.axhline(100, color='red', linestyle='--', label="Stress Threshold (100)")
        curr = move.iloc[-1]
        status = "ELEVATED (Risk Off)" if curr > 100 else "NORMAL"
        color = 'red' if curr > 100 else 'green'
        ax3.set_title(f"3. Bond Market Fear (MOVE Index): {status}", fontsize=12, weight='bold', color=color)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "MOVE Index Data Missing", ha='center')

    # 4. Sector RRG
    ax4 = fig.add_subplot(gs[3])
    rrg = calculate_rrg_metrics(prices)
    if not rrg.empty:
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)

        for i, row in rrg.iterrows():
            color = 'green' if row['Quadrant'] == 'Leading' else \
                    'blue' if row['Quadrant'] == 'Improving' else \
                    'orange' if row['Quadrant'] == 'Weakening' else 'red'
            ax4.scatter(row['RS'], row['Momentum'], color=color, s=100, alpha=0.8)
            ax4.text(row['RS'], row['Momentum'], row['Ticker'], fontsize=9, weight='bold')

        ax4.set_title("4. Sector Rotation Map (RRG Proxy)", fontsize=12, weight='bold')
        ax4.set_xlabel("Relative Strength vs SPY (Trend)", fontsize=10)
        ax4.set_ylabel("Momentum of RS (Rate of Change)", fontsize=10)
        ax4.grid(True, alpha=0.3)

        x_abs = max(abs(rrg['RS'].min()), abs(rrg['RS'].max()), 0.05) * 1.2
        y_abs = max(abs(rrg['Momentum'].min()), abs(rrg['Momentum'].max()), 0.05) * 1.2
        ax4.set_xlim(-x_abs, x_abs)
        ax4.set_ylim(-y_abs, y_abs)

        ax4.text(x_abs*0.9, y_abs*0.9, "LEADING", color='green', alpha=0.5, weight='bold', ha='right', va='top')
        ax4.text(x_abs*0.9, -y_abs*0.9, "WEAKENING", color='orange', alpha=0.5, weight='bold', ha='right', va='bottom')
        ax4.text(-x_abs*0.9, -y_abs*0.9, "LAGGING", color='red', alpha=0.5, weight='bold', ha='left', va='bottom')
        ax4.text(-x_abs*0.9, y_abs*0.9, "IMPROVING", color='blue', alpha=0.5, weight='bold', ha='left', va='top')

    plt.tight_layout()

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15) # Make room

    interp_text = "Analysis & Key Signals:\n"

    # 1. Yield Curve
    if has_yields and "10Y_Yield" in df.columns:
        curr_spread = (df["10Y_Yield"].iloc[-1] - df["2Y_Yield"].iloc[-1])
        if curr_spread < 0:
            interp_text += f"• Yield Curve (10Y-2Y): INVERTED ({curr_spread:.2f}%). Strong historical recession signal.\n"
        else:
            interp_text += f"• Yield Curve (10Y-2Y): NORMAL ({curr_spread:.2f}%). No immediate recession signal from rates.\n"

    # 2. Credit
    if "HY_Spread" in df.columns:
        curr_hy = df["HY_Spread"].dropna().iloc[-1]
        if curr_hy > 5.0:
            interp_text += f"• Credit Spreads: STRESSED ({curr_hy:.2f}%). High default risk pricing. Equity-negative.\n"
        else:
            interp_text += f"• Credit Spreads: CALM ({curr_hy:.2f}%). Corporate bond market shows no panic.\n"

    # 3. Bond Vol
    if "MOVE_Index" in df.columns:
         curr_move = df["MOVE_Index"].dropna().iloc[-1]
         if curr_move > 100:
             interp_text += f"• Bond Vol (MOVE): ELEVATED ({curr_move:.0f}). Treasury market instability poses risk to stocks.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig


def plot_macro_radar_chart(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates the Macro Regime Radar (Spider Chart)."""
    logger.info("Generating Macro Radar...")
    plt.style.use('default')

    radar_data = calculate_macro_radar(df, prices)
    if radar_data.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Insufficient Data for Radar", ha='center')
        return fig

    categories = radar_data.index.tolist()
    values = radar_data["Rank"].tolist()
    values += values[:1]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='black', size=12, weight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75], ["25", "50", "75"], color="grey", size=10)
    plt.ylim(0, 100)

    ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
    avg_rank = np.mean(values[:-1])
    fill_color = 'green' if avg_rank > 50 else 'red'
    ax.fill(angles, values, color=fill_color, alpha=0.2)

    regime_type = "EXPANSIONARY" if avg_rank > 50 else "CONTRACTIONARY"
    plt.title(f"The Shape of the Macro Regime: {regime_type}\n(1-Year Percentile Rank)", size=16, weight='bold', y=1.1)

    plt.figtext(0.5, 0.02,
                "Outer Edge (100) = Bullish/Loose/Hot | Center (0) = Bearish/Tight/Cold\n"
                "Growth: Consumer Strength | Liquidity: Money Supply | Risk: BTC/Gold\n"
                "Inflation: CPI | Rates: Bond Prices | Sentiment: Low Volatility",
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    plt.tight_layout()
    return fig


def plot_monetary_plumbing(df: pd.DataFrame) -> plt.Figure:
    """Generates Page 4: Monetary & Economic Plumbing."""
    logger.info("Generating Monetary Plumbing Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1])

    # 1. Liquidity Impulse
    ax1 = fig.add_subplot(gs[0])
    if "M2_YoY" in df.columns and "Fed_Assets_YoY" in df.columns:
        m2 = df["M2_YoY"].dropna()
        fed = df["Fed_Assets_YoY"].dropna()

        ax1.plot(m2.index, m2, color='green', label="M2 Money Supply (YoY)", linewidth=2)
        ax1.plot(fed.index, fed, color='blue', linestyle='--', label="Fed Balance Sheet (YoY)", linewidth=1.5)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.fill_between(m2.index, m2, 0, where=(m2 > 0), color='green', alpha=0.1)
        ax1.fill_between(m2.index, m2, 0, where=(m2 < 0), color='red', alpha=0.1)

        ax1.set_title("1. Liquidity Impulse: Money Supply & Fed Assets", fontsize=12, weight='bold')
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        if "M2_Velocity" in df.columns:
            m2v = df["M2_Velocity"].dropna()
            ax1_twin = ax1.twinx()
            ax1_twin.plot(m2v.index, m2v, color='orange', linestyle=':', label="M2 Velocity (Right)", linewidth=1.5)
            ax1_twin.set_ylabel("Velocity Ratio", color='orange', fontsize=10)
            ax1_twin.tick_params(axis='y', labelcolor='orange')
            ax1_twin.legend(loc="upper right")
    else:
        ax1.text(0.5, 0.5, "Liquidity Data Missing", ha='center')

    # 2. Overnight Rates Monitor (SOFR vs Fed Funds) - NEW
    ax2 = fig.add_subplot(gs[1])
    if "SOFR" in df.columns and "Fed_Funds" in df.columns:
        sofr = df["SOFR"].dropna()
        dff = df["Fed_Funds"].dropna()

        # Align
        common_idx = sofr.index.intersection(dff.index)
        sofr = sofr.loc[common_idx]
        dff = dff.loc[common_idx]

        ax2.plot(dff.index, dff, color='black', linestyle=':', label="Fed Funds Effective Rate (Policy)", linewidth=1.5)
        ax2.plot(sofr.index, sofr, color='blue', label="SOFR (Secured Overnight)", linewidth=1)

        # Stress Detection (SOFR > DFF + 5bps)
        spread = sofr - dff
        stress_dates = spread[spread > 0.05].index

        # Highlight Stress
        for date in stress_dates:
             ax2.axvline(date, color='red', alpha=0.3)

        curr_sofr = sofr.iloc[-1]
        curr_dff = dff.iloc[-1]

        status = "NORMAL"
        if curr_sofr > curr_dff + 0.05: status = "STRESS (Collateral Shortage)"
        elif curr_sofr < curr_dff - 0.10: status = "EXCESS LIQUIDITY (RRP Floor)"

        ax2.set_title(f"2. Overnight Rates Monitor (Plumbing): {curr_sofr:.2f}% vs Fed {curr_dff:.2f}% -> {status}", fontsize=12, weight='bold')
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Interpretation
        ax2.text(0.02, 0.6, "Normal: SOFR trades tight to Fed Funds.\nSpike > Fed Funds = Collateral Shortage (Repo Crisis Risk).",
                 transform=ax2.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    else:
        ax2.text(0.5, 0.5, "Overnight Rate Data Missing (SOFR/DFF)", ha='center')

    # 3. DXY
    ax3 = fig.add_subplot(gs[2])
    dxy_col = "DX-Y.NYB" if "DX-Y.NYB" in df.columns else "UUP"
    if dxy_col in df.columns:
        dxy = df[dxy_col].dropna()
        ma = dxy.rolling(200).mean()
        ax3.plot(dxy.index, dxy, color='green', label="USD Index (DXY)")
        ax3.plot(ma.index, ma, color='black', linestyle='--')

        curr = dxy.iloc[-1]
        ma_val = ma.iloc[-1]
        status = "BULLISH" if curr > ma_val else "BEARISH"
        color = 'red' if curr > ma_val else 'green'
        ax3.set_title(f"3. Global Liquidity Wrecking Ball (DXY): {status}", fontsize=12, weight='bold', color=color)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "DXY Data Missing", ha='center')

    # 4. Inflation Trend
    ax4 = fig.add_subplot(gs[3])
    if "CPI_YoY" in df.columns:
        cpi = df["CPI_YoY"].dropna()
        ax4.plot(cpi.index, cpi, color='purple', label="CPI Inflation (YoY)", linewidth=2)
        ax4.axhline(0.02, color='red', linestyle='--', label="Fed Target (2%)")
        ax4.set_title("4. Inflation Trend (CPI)", fontsize=12, weight='bold')
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        ax4.text(0.5, 0.5, "Inflation Data Missing", ha='center')

    # 5. Labor Market
    ax5 = fig.add_subplot(gs[4])
    if "Unemployment" in df.columns:
        unrate = df["Unemployment"].dropna()
        unrate_ma = unrate.rolling(12).mean()
        ax5.plot(unrate.index, unrate, color='black', label="Unemployment Rate", linewidth=2)
        ax5.plot(unrate_ma.index, unrate_ma, color='red', linestyle='--', label="12-Month Moving Avg")
        ax5.set_title("5. Labor Market Health (Unemployment)", fontsize=12, weight='bold')
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        ax5.text(0.5, 0.5, "Labor Data Missing", ha='center')

    plt.tight_layout()
    return fig


def plot_forward_models(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates Page: Forward Looking Models (Recession, Regime, Rotation, PCA, Seasonality)."""
    logger.info("Generating Forward Models Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.2, 1.2])

    # 1. Recession Probability History
    ax1 = fig.add_subplot(gs[0, :])
    if "Recession_Prob" in df.columns:
        prob = df["Recession_Prob"].dropna()
        ax1.plot(prob.index, prob, color='black', label="Recession Probability (12M Ahead)")
        ax1.fill_between(prob.index, prob, 0, color='red', alpha=0.3)
        ax1.axhline(30, color='orange', linestyle='--', label="Warning Threshold (30%)")
        ax1.axhline(50, color='red', linestyle='--', label="High Probability (>50%)")
        ax1.set_title("1. Recession Probability Model (Estrella/Mishkin Probit)", fontsize=12, weight='bold')
        ax1.set_ylabel("Probability (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
    else:
        ax1.text(0.5, 0.5, "Recession Probability Data Missing", ha='center')

    # 2. Yield Curve Spread (Input)
    ax2 = fig.add_subplot(gs[1, 0])
    if "Spread_10Y3M" in df.columns:
        spread = df["Spread_10Y3M"].dropna()
        ax2.plot(spread.index, spread, color='blue', label="10Y - 3M Treasury Spread")
        ax2.axhline(0, color='black', linewidth=1)
        ax2.fill_between(spread.index, spread, 0, where=(spread < 0), color='red', alpha=0.3, label="Inversion")
        ax2.set_title("2. The Input: 10Y-3M Yield Curve Spread", fontsize=12, weight='bold')
        ax2.set_ylabel("Spread (%)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Spread Data Missing", ha='center')

    # 3. Regime Probability (GMM)
    ax3 = fig.add_subplot(gs[1, 1])
    regime_data = calculate_regime_gmm(prices)
    if regime_data.get("current_state") != "N/A":
        labels = regime_data["labels"]
        dates = regime_data["dates"]
        ax3.scatter(dates, labels, c=labels, cmap='RdYlGn_r', s=10, alpha=0.6)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["State 0", "State 1", "State 2"])
        ax3.set_title(f"3. Market Regimes (GMM Clustering)\nCurrent: {regime_data['current_state']}", fontsize=12, weight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Regime Data Missing", ha='center')

    # 4. Sector Rotation Matrix
    ax4 = fig.add_subplot(gs[2, 0])
    rot_data = predict_sector_rotation(prices)
    if rot_data["current_leader"] != "N/A":
        sectors = ["XLI", "XLB", "XLU", "XLF", "XLK", "XLE", "XLV", "XLC", "XLY", "XLP"]
        valid_sectors = [s for s in sectors if s in prices.columns]

        # Recalculate basic transitions for bar chart
        monthly_prices = prices[valid_sectors].resample('ME').last()
        monthly_rets = monthly_prices.pct_change().infer_objects(copy=False).dropna()
        leaders = monthly_rets.idxmax(axis=1)
        curr_leader = rot_data["current_leader"]

        next_counts = {}
        for prev, curr in zip(leaders[:-1], leaders[1:]):
            if prev == curr_leader:
                next_counts[curr] = next_counts.get(curr, 0) + 1

        if next_counts:
            total = sum(next_counts.values())
            sorted_counts = sorted(next_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [x[0] for x in sorted_counts]
            vals = [x[1]/total for x in sorted_counts]

            ax4.bar(labels, vals, color='purple', alpha=0.7)
            ax4.set_title(f"4. Rotation Clock: Next Leader Prob\n(Given Best Performer: {curr_leader})", fontsize=12, weight='bold')
            ax4.set_ylabel("Probability")
            ax4.grid(True, axis='y', alpha=0.3)
        else:
             ax4.text(0.5, 0.5, f"No historical precedents for leader: {curr_leader}", ha='center')
    else:
        ax4.text(0.5, 0.5, "Rotation Data Missing", ha='center')

    # 5. Systemic Risk (PCA)
    ax5 = fig.add_subplot(gs[2, 1])
    pca_data = calculate_systemic_risk_pca(prices)
    if not pca_data["history"].empty:
        hist = pca_data["history"]
        ax5.plot(hist.index, hist, color='red', label="Absorption Ratio")
        ax5.axhline(0.75, color='black', linestyle='--', label="Critical (>75%)")
        ax5.axhline(0.65, color='orange', linestyle='--', label="Elevated (>65%)")
        ax5.set_title(f"5. Systemic Risk Monitor (PCA)\nStatus: {pca_data['status']}", fontsize=12, weight='bold')
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1.0)
    else:
        ax5.text(0.5, 0.5, "PCA Data Missing", ha='center')

    # 6. Seasonality
    ax6 = fig.add_subplot(gs[3, :])
    seas_data = calculate_seasonality(prices)
    if seas_data["curr_month"] != "N/A":
        # Recalculate monthly seasonality
        spy = prices["SPY"]
        spy_monthly = spy.resample('ME').last().pct_change().infer_objects(copy=False).dropna()
        df_m = pd.DataFrame({"Ret": spy_monthly})
        df_m["Month"] = df_m.index.month
        monthly_stats = df_m.groupby("Month")["Ret"].mean()

        import calendar
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        colors = ['green' if x > 0 else 'red' for x in monthly_stats]

        bars = ax6.bar(month_names, monthly_stats, color=colors, alpha=0.5)

        curr_m = dt.date.today().month
        next_m = (curr_m % 12) + 1

        # Highlight
        if 0 <= curr_m-1 < 12:
            bars[curr_m-1].set_alpha(1.0)
            bars[curr_m-1].set_edgecolor('black')
            bars[curr_m-1].set_linewidth(2)
        if 0 <= next_m-1 < 12:
            bars[next_m-1].set_alpha(1.0)
            bars[next_m-1].set_edgecolor('blue')
            bars[next_m-1].set_linewidth(2)

        ax6.set_title(f"6. Seasonal Cycle Forecast (Avg Monthly Return)\nHighlight: {seas_data['curr_month']} (Black) & {seas_data['next_month']} (Blue)", fontsize=12, weight='bold')
        ax6.axhline(0, color='black', linewidth=1)
        ax6.grid(True, axis='y', alpha=0.3)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    else:
        ax6.text(0.5, 0.5, "Seasonality Data Missing", ha='center')

    plt.tight_layout()

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.12)

    interp_text = "Forward Looking Signals:\n"

    # 1. Recession Prob
    if "Recession_Prob" in df.columns:
        curr_prob = df["Recession_Prob"].iloc[-1]
        if curr_prob > 30:
            interp_text += f"• Recession Model: WARNING ({curr_prob:.1f}%). Probability exceeds safety threshold.\n"
        else:
            interp_text += f"• Recession Model: LOW RISK ({curr_prob:.1f}%). Yield curve does not signal imminent downturn.\n"

    # 2. Regime
    interp_text += "• Market Regime: Verify with GMM plot. (State 0 = Bull, State 1 = Volatile/Bear).\n"

    # 3. Seasonality
    curr_m = dt.date.today().month
    import calendar
    m_name = calendar.month_abbr[curr_m]

    interp_text += f"• Seasonality ({m_name}): Check bar chart. Historic avg return provides bias.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig


def plot_predictive_models_page(df: pd.DataFrame, internals: Dict, recession_prob: pd.Series) -> plt.Figure:
    """Page 11: Predictive Analytics (Recession & Internals)."""
    logger.info("Generating Predictive Models Page...")
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("PREDICTIVE ANALYTICS: MACRO & MARKET INTERNALS", fontsize=16, weight='bold', y=0.98)

    # Grid: 2 Rows. Top = Recession. Bottom = Internals.
    gs = fig.add_gridspec(2, 1, hspace=0.3)

    # --- Panel 1: Recession Probability ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Combine Model & actual Recessions if available?
    # We'll just plot the Probit Probability
    if not recession_prob.empty:
        prob = recession_prob.iloc[-1260:] # Last 5 years
        ax1.plot(prob.index, prob.values, color='red', linewidth=2, label="Recession Prob (12M Fwd)")
        ax1.fill_between(prob.index, prob.values, 0, color='red', alpha=0.3)

        # Add Threshold Line
        ax1.axhline(30, color='black', linestyle='--', alpha=0.5, label="Warning Threshold (30%)")

        curr_prob = prob.iloc[-1]
        ax1.set_title(f"NY Fed Recession Probability Model (Current: {curr_prob:.1f}%)", fontsize=12, weight='bold')
        ax1.set_ylabel("Probability (%)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Annotate
        if curr_prob > 30:
            ax1.text(prob.index[-1], curr_prob, " HIGH RISK", color='red', weight='bold')
    else:
        ax1.text(0.5, 0.5, "Data Unavailable", ha='center')

    # --- Panel 2: Market Internals (Leading Indicators) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Market Internals: Leading Ratios (Normalized)", fontsize=12, weight='bold')

    for name, data in internals.items():
        series = data["Series"].dropna().iloc[-252:] # Last 1 Year
        if series.empty: continue

        # Normalize to start at 0%
        norm_series = (series / series.iloc[0] - 1) * 100

        if name == "Defensive":
             # Invert Defensive for visualization? No, let's keep as is but specific color
             ax2.plot(norm_series.index, norm_series.values, label=name, linestyle='--', alpha=0.7)
        else:
             ax2.plot(norm_series.index, norm_series.values, label=name, linewidth=1.5)

    ax2.set_ylabel("Change (%)")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15, top=0.93)

    interp_text = "Predictive Analytics:\n"
    # Recession
    if not recession_prob.empty:
         rp = recession_prob.iloc[-1]
         if rp > 30: interp_text += f"• Recession Risk: HIGH ({rp:.1f}%). Yield curve signals economic contraction ahead.\n"
         else: interp_text += f"• Recession Risk: LOW ({rp:.1f}%). No immediate curve-driven signal.\n"

    # Internals
    interp_text += "• Leading Indicators: Watch for 'Cyclical' vs 'Defensive' divergence. If Defensives lead, risk-off/recession is likely."

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

    # Add Explainer Text box
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. RECESSION MODEL: Uses the Yield Curve (10Y-3M) to predict recession chance in next 12 months.\n"
        "   - >30% = Warning. >50% = High Probability.\n"
        "2. MARKET INTERNALS (Leading Indicators):\n"
        "   - Risk Appetite (XLY/XLP): Rising means investors prefer Growth/Cyclicals (Bullish).\n"
        "   - Breadth (RSP/SPY): Rising means broad participation (Healthy).\n"
        "   - Credit (HYG/IEF): Rising means Credit Markets are ignoring risk (Bullish/Complacent)."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) # Make room for text
    return fig
