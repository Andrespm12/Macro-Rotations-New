"""
Visualization Functions (Matplotlib).
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from typing import Dict, List
from analytics.rotations import calculate_rrg_metrics, predict_sector_rotation, calculate_seasonality
from analytics.macro_models import calculate_macro_radar, calculate_regime_gmm
from analytics.quant import calculate_strategy_performance, calculate_systemic_risk_pca, fit_garch, simulate_gbm, calculate_greeks, black_scholes_merton, calculate_antifragility_metrics
from analytics.stochastic import simulate_heston, simulate_merton_jump, simulate_hawkes_intensity
from analytics.econometrics import fit_markov_regime_switching, fit_ou_process

def run_backtest_plot(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Runs a vectorised backtest for multiple strategies and plots the results."""
    print("   Running Backtest & Projections...")
    
    metrics, curves = calculate_strategy_performance(df, prices)
    if curves is None or curves.empty:
        print("   Backtest failed: Missing assets.")
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
        rets = series.pct_change().dropna()
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

def plot_risk_macro_dashboard(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates Page 3: Macro Risk & Sector Rotation."""
    print("   Generating Risk & Macro Page...")
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
    return fig

def plot_macro_radar_chart(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates the Macro Regime Radar (Spider Chart)."""
    print("   Generating Macro Radar...")
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
    print("   Generating Monetary Plumbing Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
    
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

    # 2. DXY
    ax2 = fig.add_subplot(gs[1])
    dxy_col = "DX-Y.NYB" if "DX-Y.NYB" in df.columns else "UUP"
    if dxy_col in df.columns:
        dxy = df[dxy_col].dropna()
        ma = dxy.rolling(200).mean()
        ax2.plot(dxy.index, dxy, color='green', label="USD Index (DXY)")
        ax2.plot(ma.index, ma, color='black', linestyle='--')
        
        curr = dxy.iloc[-1]
        ma_val = ma.iloc[-1]
        status = "BULLISH" if curr > ma_val else "BEARISH"
        color = 'red' if curr > ma_val else 'green'
        ax2.set_title(f"2. Global Liquidity Wrecking Ball (DXY): {status}", fontsize=12, weight='bold', color=color)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "DXY Data Missing", ha='center')

    # 3. Inflation Trend
    ax3 = fig.add_subplot(gs[2])
    if "CPI_YoY" in df.columns:
        cpi = df["CPI_YoY"].dropna()
        ax3.plot(cpi.index, cpi, color='purple', label="CPI Inflation (YoY)", linewidth=2)
        ax3.axhline(0.02, color='red', linestyle='--', label="Fed Target (2%)")
        ax3.set_title("3. Inflation Trend (CPI)", fontsize=12, weight='bold')
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    else:
        ax3.text(0.5, 0.5, "Inflation Data Missing", ha='center')
        
    # 4. Labor Market
    ax4 = fig.add_subplot(gs[3])
    if "Unemployment" in df.columns:
        unrate = df["Unemployment"].dropna()
        unrate_ma = unrate.rolling(12).mean()
        ax4.plot(unrate.index, unrate, color='black', label="Unemployment Rate", linewidth=2)
        ax4.plot(unrate_ma.index, unrate_ma, color='red', linestyle='--', label="12-Month Moving Avg")
        ax4.set_title("4. Labor Market Health (Unemployment)", fontsize=12, weight='bold')
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    else:
        ax4.text(0.5, 0.5, "Labor Data Missing", ha='center')

    plt.tight_layout()
    return fig

def plot_forward_models(df: pd.DataFrame, prices: pd.DataFrame) -> plt.Figure:
    """Generates Page: Forward Looking Models (Recession, Regime, Rotation, PCA, Seasonality)."""
    print("   Generating Forward Models Page...")
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
        monthly_rets = monthly_prices.pct_change().dropna()
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
        spy_monthly = spy.resample('ME').last().pct_change().dropna()
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
    return fig

def plot_quant_lab_dashboard(prices: pd.DataFrame) -> plt.Figure:
    """Generates Page: Quant Lab (Vol, Monte Carlo, Greeks)."""
    print("   Generating Quant Lab Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Volatility Regime
    ax1 = fig.add_subplot(gs[0, :])
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().dropna()
        
        # GARCH Fit
        try:
            vol_garch, _ = fit_garch(spy_ret.values * 100)
            vol_garch = vol_garch / 100 * np.sqrt(252)
            garch_series = pd.Series(vol_garch, index=spy_ret.index)
        except:
            garch_series = pd.Series(0, index=spy_ret.index)
            
        vol_hist = spy_ret.rolling(21).std() * np.sqrt(252)
        lookback = 252
        
        ax1.plot(garch_series.index[-lookback:], garch_series.iloc[-lookback:], label="GARCH(1,1) Est", color='purple', linewidth=2)
        ax1.plot(vol_hist.index[-lookback:], vol_hist.iloc[-lookback:], label="Realized (21D)", color='orange', alpha=0.7)
        ax1.set_title("SPY Volatility Regime: GARCH Model vs Realized", fontsize=12, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "SPY Data Missing", ha='center')

    # 2. Monte Carlo
    ax2 = fig.add_subplot(gs[1, 0])
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().dropna()
        S0 = prices["SPY"].iloc[-1]
        mu = spy_ret.mean() * 252
        sigma_sim = spy_ret.std() * np.sqrt(252)
        T_sim = 30/252.0 # 30 Days
        dt_sim = 1/252.0
        n_paths = 100
        
        try:
            time, paths = simulate_gbm(S0, mu, sigma_sim, T_sim, dt_sim, n_paths)
            for i in range(n_paths):
                ax2.plot(time*252, paths[i], color='cyan', alpha=0.1)
            ax2.plot(time*252, paths.mean(axis=0), color='blue', linewidth=2, label="Mean Path")
            ax2.set_title(f"Monte Carlo: SPY 30-Day Projection ({n_paths} Paths)", fontsize=12, weight='bold')
            ax2.set_xlabel("Days Ahead")
            ax2.set_ylabel("Price")
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(0.5, 0.5, f"Sim Error: {e}", ha='center')
    else:
        ax2.text(0.5, 0.5, "Data Missing", ha='center')

    # 3. ATM Greeks
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    if "SPY" in prices.columns:
        S = prices["SPY"].iloc[-1]
        K = S
        T = 30/365.0
        r = 0.045
        sigma = 0.15 # Fallback or calc
        if "sigma_sim" in locals(): sigma = sigma_sim
        
        greeks = calculate_greeks(S, K, T, r, sigma, "call")
        bsm_price = black_scholes_merton(S, K, T, r, sigma, "call")
        
        greeks_data = [
            ["Metric", "Value"],
            ["ATM Call Price (30D)", f"${bsm_price:.2f}"],
            ["Delta", f"{greeks.get('Delta', 0):.3f}"],
            ["Gamma", f"{greeks.get('Gamma', 0):.4f}"],
            ["Theta (Daily)", f"{greeks.get('Theta', 0):.3f}"],
            ["Vega (1%)", f"{greeks.get('Vega', 0):.3f}"],
            ["Rho", f"{greeks.get('Rho', 0):.3f}"]
        ]
        
        table = ax3.table(cellText=greeks_data, loc='center', cellLoc='center', colWidths=[0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        ax3.set_title("SPY ATM Option Greeks (Theoretical)", fontsize=12, weight='bold')
    else:
        ax3.text(0.5, 0.5, "Data Missing", ha='center')

    plt.tight_layout()
    return fig

def plot_alpha_factors_page(prices: pd.DataFrame, macro: pd.DataFrame, alpha_data: Dict) -> plt.Figure:
    """Generates Page 6: Institutional Alpha Factors."""
    print("   Generating Alpha Factors Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 14))
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
    return fig

def plot_cross_asset_page(prices: pd.DataFrame, corr_data: Dict) -> plt.Figure:
    """Generates Page 9: Cross-Asset Regime."""
    print("   Generating Cross-Asset Correlation Page...")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(14, 14))
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
    return fig

def plot_efficient_frontier_page(optimization_data: Dict) -> plt.Figure:
    """Generates Page 10: Portfolio Optimization Lab."""
    print("   Generating Efficient Frontier Page...")
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
    return fig

def plot_predictive_models_page(df: pd.DataFrame, internals: Dict, recession_prob: pd.Series) -> plt.Figure:
    """Page 11: Predictive Analytics (Recession & Internals)."""
    print("   Generating Predictive Models Page...")
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
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
        
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

def plot_monte_carlo_cone(prices: pd.DataFrame, ticker: str = "SPY", days: int = 60, n_sims: int = 1000) -> plt.Figure:
    """Page 12: Quant Lab Simulation (Brownian Motion Cone)."""
    print(f"   Generating Monte Carlo Cone for {ticker}...")
    
    if ticker not in prices.columns: return None
    
    # 1. Calibrate Model
    series = prices[ticker].dropna()
    rets = series.pct_change().dropna()
    
    S0 = series.iloc[-1]
    mu = rets.mean() * 252
    sigma = rets.std() * np.sqrt(252)
    
    # 2. Simulate
    T = days / 252.0
    dt_step = 1 / 252.0
    
    time_sim, paths = simulate_gbm(S0, mu, sigma, T, dt_step, n_sims)
    
    # 3. Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"QUANT LAB: MONTE CARLO SIMULATION ({ticker})", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    
    # Historical Context (6 Months)
    hist_window = 126
    hist_dates = series.index[-hist_window:]
    hist_prices = series.values[-hist_window:]
    
    ax1.plot(hist_dates, hist_prices, color='black', linewidth=2, label="Historical Price")
    
    # Generate Future Dates
    last_date = series.index[-1]
    future_dates = [last_date + dt.timedelta(days=i) for i in range(len(time_sim))] 
    # Note: simulate_gbm returns time steps 0..T. 0 is today.
    # We need to map 'business days' ideally, but T+timedelta is fine for viz.
    # Let's use business days logic for cleaner x-axis if possible, or just standard days
    future_dates = pd.date_range(start=last_date, periods=len(time_sim), freq='B')
    
    # Plot Paths (First 100)
    for i in range(min(100, n_sims)):
        ax1.plot(future_dates, paths[i, :], color='gray', alpha=0.1, linewidth=0.5)
        
    # Percentiles
    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    ax1.plot(future_dates, p50, color='blue', linewidth=2, label="Median Path (P50)")
    ax1.plot(future_dates, p5, color='red', linestyle='--', linewidth=1.5, label="95% Confidence Interval")
    ax1.plot(future_dates, p95, color='red', linestyle='--', linewidth=1.5)
    
    ax1.fill_between(future_dates, p5, p95, color='blue', alpha=0.1)
    
    ax1.set_title(f"Geometric Brownian Motion: {days}-Day Forecast Cone", fontsize=12, weight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Annotate Final Range
    final_p5 = p5[-1]
    final_p95 = p95[-1]
    ax1.text(future_dates[-1], final_p95, f"${final_p95:.0f}", color='red', va='bottom')
    ax1.text(future_dates[-1], final_p5, f"${final_p5:.0f}", color='red', va='top')
    
    # --- Panel 2: Distribution of Returns ---
    ax2 = fig.add_subplot(gs[1])
    final_prices = paths[:, -1]
    rets_sim = (final_prices / S0) - 1
    
    ax2.hist(rets_sim, bins=50, color='navy', alpha=0.7, density=True)
    ax2.axvline(0, color='black', linestyle='-')
    
    # Stats
    win_prob = (rets_sim > 0).mean()
    exp_val = rets_sim.mean()
    
    ax2.set_title(f"Distribution of Simulated Returns (Win Probability: {win_prob:.1%})", fontsize=12, weight='bold')
    ax2.set_xlabel("Return (%)")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, alpha=0.3)
    
    # Text Box
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. THE CONE: Projecting future price paths using Geometric Brownian Motion (Random Walk + Drift).\n"
        "2. PROBABILITY BANDS: 68% (1 Sigma) and 95% (2 Sigma) confidence intervals.\n"
        "3. USE CASE: Setting realistic profit targets (Upper Band) and stop-losses (Lower Band) based on volatility.\n"
        "4. PARAMETERS: Drift = Expected Return, Volatility = Average Risk."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    return fig

def plot_stochastic_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 13: Stochastic Volatility & Regime Switching."""
    print(f"   Generating Stochastic Models Page for {ticker}...")
    
    if ticker not in prices.columns: return None
    
    # 1. Heston Calib (Simplified)
    series = prices[ticker].dropna()
    rets = series.pct_change().dropna()
    S0 = series.iloc[-1]
    
    # Simplified manual params for visual demo
    # Real Heston calibrating is complex optimization, we use illustrative parameters
    v0 = rets.var() * 252 # Annualized variance
    mu = 0.08
    kappa = 2.0  # Mean reversion speed
    theta = 0.04 # Long run variance (20% vol squared)
    xi = 0.3     # Vol of Vol
    rho = -0.7   # Leverage effect
    T = 1.0      # 1 Year
    n_sims = 100 
    
    time_sim, S, v = simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, 252, n_sims)
    
    # 2. HMM Regime Fit
    hmm_res = fit_markov_regime_switching(series.pct_change().dropna(), k_regimes=2)
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: ADVANCED STOCHASTIC MODELS", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.4)
    
    # Panel 1: Heston Price Paths
    ax1 = fig.add_subplot(gs[0])
    # Show history + future
    # Just show future paths for clarity
    future_dates = pd.date_range(start=series.index[-1], periods=len(time_sim), freq='B')
    
    for i in range(min(50, n_sims)): # Plot 50 paths
        ax1.plot(future_dates, S[i, :], color='blue', alpha=0.15, linewidth=0.5)
        
    ax1.set_title(f"1. Heston Stochastic Volatility Model (1-Year Simulation)", fontsize=12, weight='bold')
    ax1.set_ylabel("Price ($)")
    
    # Panel 2: Heston Volatility Paths
    ax2 = fig.add_subplot(gs[1])
    for i in range(min(50, n_sims)):
        vol_path = np.sqrt(v[i, :]) * 100 # Convert variance -> vol %
        ax2.plot(future_dates, vol_path, color='orange', alpha=0.15, linewidth=0.5)
        
    ax2.set_title("2. Simulated Volatility Paths (Stochastic Process)", fontsize=12, weight='bold')
    ax2.set_ylabel("Volatility (%)")
    
    # Panel 3: Markov Regime Probabilities
    ax3 = fig.add_subplot(gs[2])
    probs = hmm_res.get("probs", pd.DataFrame())
    
    if not probs.empty:
        # Plot only last 500 days for visibility
        subset = probs.iloc[-500:]
        # Stacked area
        ax3.stackplot(subset.index, subset.T.values, labels=subset.columns, alpha=0.6, colors=['green', 'red'])
        
        curr_regime = "Unknown"
        if not subset.empty:
            curr_regime = subset.iloc[-1].idxmax()
            
        ax3.set_title(f"3. Markov Regime Switching (Current: {curr_regime})", fontsize=12, weight='bold')
        ax3.set_ylabel("Probability")
        ax3.legend(loc='upper left')
    else:
        ax3.text(0.5, 0.5, "HMM Fit Failed (Insufficient Data)", ha='center')
        
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. HESTON MODEL: Simulates Stochastic Volatility (Vol is not constant, it's a process).\n"
        "2. REGIME SWITCHING (HMM): Detects 'Calm' (Low Vol) vs 'Turbulent' (High Vol) market states.\n"
        "3. USE CASE: Calibrating options strategies. Buy Volatility when regime switches to 'Turbulent'."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    return fig

def plot_mean_reversion_page(prices: pd.DataFrame) -> plt.Figure:
    """Page 14: Mean Reversion (Ornstein-Uhlenbeck)."""
    print("   Generating Mean Reversion Page...")
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: MEAN REVERSION (OU PROCESS)", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # --- PAIR 1: YIELD CURVE (IEF vs SHY) ---
    ax1 = fig.add_subplot(gs[0])
    pair1_name = "Yield Curve Spread (IEF 7-10Y - SHY 1-3Y)"
    if "IEF" in prices.columns and "SHY" in prices.columns:
        s1 = prices["IEF"]
        s2 = prices["SHY"]
        # Normalize roughly or just take raw spread? Prices are different scales (approx $94 vs $81).
        # Better: Log Spread or Ratio. Let's use Ratio for stationarity.
        spread = np.log(s1 / s2)
        
        # Fit OU
        ou_params = fit_ou_process(spread)
        theta = ou_params.get("theta", 0)
        mu = ou_params.get("mu", 0)
        sigma = ou_params.get("sigma", 0)
        hl = ou_params.get("half_life", 0)
        
        # Plot
        ax1.plot(spread.index, spread.values, color='black', label="Log Spread (IEF/SHY)")
        ax1.axhline(mu, color='blue', linestyle='--', label=f"Long Run Mean ({mu:.4f})")
        
        # Sigma Bands
        if not np.isnan(sigma) and theta > 0:
            # Stationary variance = sigma^2 / (2*theta)
            std_dev = sigma / np.sqrt(2*theta)
            ax1.axhline(mu + 2*std_dev, color='red', linestyle=':', label="+2 Sigma")
            ax1.axhline(mu - 2*std_dev, color='red', linestyle=':', label="-2 Sigma")
            
            # Current Z-Score
            curr = spread.iloc[-1]
            z_score = (curr - mu) / std_dev
            ax1.set_title(f"1. {pair1_name}\nMean Reversion Speed (Theta): {theta:.2f} | Half-Life: {hl:.1f} Days | Current Z-Score: {z_score:.2f}", fontsize=12, weight='bold')
            
            # Annotate Trade Signal
            if z_score > 2.0:
                 ax1.text(spread.index[-1], curr, " OVERBOUGHT (Short Spread)", color='red', weight='bold')
            elif z_score < -2.0:
                 ax1.text(spread.index[-1], curr, " OVERSOLD (Long Spread)", color='green', weight='bold')
        
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Data Missing (IEF/SHY)", ha='center')
        
    # --- PAIR 2: USD vs YEN (UUP vs FXY) ---
    ax2 = fig.add_subplot(gs[1])
    pair2_name = "USD/JPY Proxy (UUP - FXY)" # Spread between Dollar ETF and Yen ETF
    if "UUP" in prices.columns and "FXY" in prices.columns:
        # FXY is Yen inverted (Yen strength). UUP is Dollar strength.
        # Just use log ratio again.
        spread2 = np.log(prices["UUP"] / prices["FXY"])
        
        # Fit OU
        ou_params2 = fit_ou_process(spread2)
        theta2 = ou_params2.get("theta", 0)
        mu2 = ou_params2.get("mu", 0)
        sigma2 = ou_params2.get("sigma", 0)
        hl2 = ou_params2.get("half_life", 0)
        
        ax2.plot(spread2.index, spread2.values, color='purple', label="Log Spread (UUP/FXY)")
        ax2.axhline(mu2, color='blue', linestyle='--', label="Mean")
        
        if not np.isnan(sigma2) and theta2 > 0:
             std_dev2 = sigma2 / np.sqrt(2*theta2)
             ax2.axhline(mu2 + 2*std_dev2, color='red', linestyle=':')
             ax2.axhline(mu2 - 2*std_dev2, color='red', linestyle=':')
             
             curr2 = spread2.iloc[-1]
             z2 = (curr2 - mu2) / std_dev2
             
             ax2.set_title(f"2. {pair2_name}\nMean Reversion Speed (Theta): {theta2:.2f} | Half-Life: {hl2:.1f} Days | Current Z-Score: {z2:.2f}", fontsize=12, weight='bold')
        else:
             ax2.set_title(f"2. {pair2_name} (Trending / Non-Stationary)", fontsize=12, weight='bold')
             
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Data Missing (UUP/FXY)", ha='center')

    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. ORNSTEIN-UHLENBECK (OU): Models Mean-Reverting assets (Spreads, Pairs).\n"
        "2. Z-SCORE: Measures distance from the mean. > 2.0 is statistically stretched (95% confidence).\n"
        "3. SIGNAL: High Z-Score + High Mean Reversion Speed (Theta) = Strong probability of snap-back."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
             
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20) # Increase bottom margin
    return fig

def plot_microstructure_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 15: Jump Diffusion & Hawkes Microstructure."""
    print("   Generating Microstructure & Jumps Page...")
    
    if ticker not in prices.columns: return None
    
    idx = prices.index
    series = prices[ticker].dropna()
    S0 = series.iloc[-1]
    
    # --- 1. Merton Simulation ---
    # Scenarios: "Fat Tail" risks
    mu = 0.08
    sigma = 0.15
    # Crash Params
    lambda_jump = 2.0  # 2 jumps per year on average
    mu_jump = -0.10    # Average jump is -10%
    sigma_jump = 0.05  # Std dev of jump
    T = 1.0
    n_sims = 100
    
    time, S = simulate_merton_jump(S0, mu, sigma, T, 252, n_sims, lambda_jump, mu_jump, sigma_jump)
    
    # --- 2. Hawkes Simulation ---
    # Simulate Order Flow / Volatility Clustering
    # mu (base) = 1.0 events/day
    # alpha (excitation) = 0.8
    # beta (decay) = 1.2
    h_mu = 1.0
    h_alpha = 0.8
    h_beta = 1.2
    
    time_h, intensity, events = simulate_hawkes_intensity(h_mu, h_alpha, h_beta, 100, 1000)
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: JUMPS & MARKET MICROSTRUCTURE", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Panel 1: Merton Jumps
    ax1 = fig.add_subplot(gs[0])
    future_dates = pd.date_range(start=idx[-1], periods=len(time), freq='B')
    
    for i in range(min(50, n_sims)):
        ax1.plot(future_dates, S[i, :], color='blue', alpha=0.1, linewidth=0.5)
        
    ax1.set_title(f"1. Merton Jump Diffusion (Simulating 'Fat Tail' Risks)\nParams: {lambda_jump} Jumps/Yr, Mean Size {mu_jump:.0%} (Crash Scenarios)", fontsize=12, weight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Hawkes Intensity
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_h, intensity, color='purple', linewidth=1.5, label="Intensity (Event Arrival Rate)")
    
    # Mark events
    # events array is timepoints
    # We used direct simulation in stochastics returning simple arrays
    # if events is array of times:
    if len(events) > 0:
        y_ev = np.full_like(events, 0.5) # Plot dots at bottom? Or verify return type
        # My stochastics.simulate_hawkes returned time, intensity, event_times (array)
        # Check signature: returns time, intensity, event_times
        # So "events" here is event_times.
        ax2.scatter(events, np.full_like(events, intensity.min()), color='red', marker='|', alpha=0.6, label="Event (Order/shock)")
        
    ax2.set_title("2. Hawkes Process (Self-Exciting Clustering)\nModeling 'Volatility Clustering' or 'Flash Crashes'", fontsize=12, weight='bold')
    ax2.set_ylabel("Intensity (lambda)")
    ax2.set_xlabel("Time (Arbitrary Units)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. MERTON JUMPS: Models sudden 'Crashes' (Jumps) that normal models miss. Shows 'Gap Risk'.\n"
        "2. HAWKES PROCESS: Models 'Self-Excitement' (Clustering). One shock triggers others (Feedback Loops).\n"
        "3. APPLICATION: Stress-testing portfolios against 'Black Swans' and 'Flash Crashes'."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20) # Increase bottom margin
    return fig

def plot_antifragility_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 16: Taleb Anti-Fragility & Tail Risk."""
    print("   Generating Anti-Fragility Analysis Page...")
    
    if ticker not in prices.columns: return None
    
    series = prices[ticker].dropna()
    returns = series.pct_change().dropna()
    
    metrics = calculate_antifragility_metrics(returns)
    skew = metrics.get("skew", 0)
    kurt = metrics.get("kurtosis", 0)
    taleb_ratio = metrics.get("taleb_ratio", 0)
    status = metrics.get("status", "N/A")
    
    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: ANTI-FRAGILITY & BLACK SWAN VALIDATION", fontsize=16, weight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)
    
    # Panel 1: Return Distribution (Fat Tails)
    ax1 = fig.add_subplot(gs[0, :]) # Top full width
    
    import seaborn as sns
    sns.histplot(returns, bins=100, kde=True, stat="density", color="blue", alpha=0.3, ax=ax1, label="Actual Distribution")
    
    # Normal Distribution Overlay
    mu, std = norm.fit(returns)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'r--', linewidth=2, label=f"Normal Dist (Gaussian)")
    
    ax1.set_title(f"1. Tail Risk Analysis (Actual vs Normal)\nSkew: {skew:.2f} (Target > 0) | Kurtosis: {kurt:.2f} (Fat Tails)", fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Taleb Ratio Visual (Upside vs Downside Vol)
    ax2 = fig.add_subplot(gs[1, 0])
    
    upside = returns[returns > 0]
    downside = returns[returns < 0] # Make positive for comparison
    
    ax2.boxplot([upside, abs(downside)], labels=["Upside Returns", "Downside Risk (Abs)"], patch_artist=True, 
                boxprops=dict(facecolor="lightblue"))
    
    ax2.set_title(f"2. Asymmetry Analysis (Taleb Ratio)\nRatio: {taleb_ratio:.2f} (Target > 1.1)", fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Fragility Gauge (Status)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Draw simple text gauge
    color = "green" if "ANTI-FRAGILE" in status else "red" if "FRAGILE" in status else "orange"
    
    ax3.text(0.5, 0.7, "PORTFOLIO CLASSIFICATION:", ha='center', fontsize=12)
    ax3.text(0.5, 0.5, status, ha='center', fontsize=16, weight='bold', color=color, 
             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=1'))
             
    ax3.text(0.5, 0.3, f"The 'Turkey' Score: {metrics.get('turkey_score', 0):.2f}\n(Hidden Tail Risk)", ha='center', fontsize=10)

    # Footer (Enhanced Commentary)
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. TAIL RISK: We want the blue distribution to shift RIGHT (Positive Skew). Fat left tails indicate crash risk.\n"
        "2. TALEB RATIO: Measures payoff assymetry. Ratio > 1.1 means upside volatility > downside volatility (Good).\n"
        "3. TURKEY SCORE: A high negative score means steady small gains but massive hidden tail risk (like a Turkey before Thanksgiving).\n"
        "4. GOAL: We seek 'Anti-Fragility' -> Positioning that benefits from volatility and disorder (Convex Payoffs)."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    return fig
