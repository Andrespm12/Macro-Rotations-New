"""Quantitative lab, stochastic models, and scenario analysis charts."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm
from typing import Dict

from analytics.quant import fit_garch, simulate_gbm, calculate_greeks, black_scholes_merton, calculate_antifragility_metrics
from analytics.stochastic import simulate_heston, simulate_merton_jump, simulate_hawkes_intensity
from analytics.econometrics import fit_markov_regime_switching, fit_ou_process
from analytics.scenarios import calculate_move_probabilities, generate_contingencies
from core.logger import get_logger

logger = get_logger(__name__)

def plot_quant_lab_dashboard(prices: pd.DataFrame) -> plt.Figure:
    """Generates Page: Quant Lab (Vol, Monte Carlo, Greeks)."""
    logger.info("Generating Quant Lab Page...")
    plt.style.use('default')

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. Volatility Regime
    ax1 = fig.add_subplot(gs[0, :])
    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().infer_objects(copy=False).dropna()

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
        spy_ret = prices["SPY"].pct_change().infer_objects(copy=False).dropna()
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

    # --- Interpretation Box (Bottom) ---
    fig.subplots_adjust(bottom=0.15)

    interp_text = "Quant Lab Insights:\n"

    # 1. Vol
    if "SPY" in prices.columns:
        curr_vol = vol_hist.iloc[-1]
        interp_text += f"• Volatility Regime (21D Realized): {curr_vol:.1f}%. "
        if curr_vol < 12: interp_text += "Low Volatility (Complacency/Bull Trend).\n"
        elif curr_vol > 25: interp_text += "High Volatility (Fear/Crash Risk).\n"
        else: interp_text += "Normal Volatility.\n"

    # 2. Monte Carlo
    if "SPY" in prices.columns:
        mean_path = paths.mean(axis=0)[-1]
        chg = (mean_path/S0 - 1) * 100
        interp_text += f"• Monte Carlo Projection (Mean): {chg:.1f}% expected return over 30 days.\n"

    # 3. Greeks
    interp_text += "• Option Greeks: Delta measures directional exposure. Vega measures sensitivity to volatility spikes.\n"

    fig.text(0.05, 0.02, interp_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
    return fig

def plot_monte_carlo_cone(prices: pd.DataFrame, ticker: str = "SPY", days: int = 60, n_sims: int = 1000) -> plt.Figure:
    """Page 12: Quant Lab Simulation (Brownian Motion Cone)."""
    logger.info("Generating Monte Carlo Cone for %s...", ticker)

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

    # --- Interpretation Box ---
    interp_text = (
        "INTERPRETATION GUIDE:\n"
        "• THE CONE: Shows the range of probable price paths based on drift (trend) and diffusion (volatility).\n"
        "• MEDIAN PATH (Blue): The 'Base Case' projection.\n"
        "• 95% CONFIDENCE (Light Blue): Outlier scenarios. If price hits the edge, it is empirically overextended.\n"
        "• USE CASE: Validates price targets. If your target is outside the cone, it requires an extreme event."
    )
    fig.text(0.05, 0.02, interp_text, fontsize=9, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))

    return fig

def plot_stochastic_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 13: Stochastic Volatility & Regime Switching."""
    logger.info("Generating Stochastic Models Page for %s...", ticker)

    if ticker not in prices.columns: return None
    series = prices[ticker].dropna()

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"STOCHASTIC MODELLING & REGIME DETECTION ({ticker})", fontsize=16, weight='bold', y=0.98)

    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.4)

    # 1. Heston Calib (Simplified)
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
    logger.info("Generating Mean Reversion Page...")

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
    logger.info("Generating Microstructure & Jumps Page...")

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
    logger.info("Generating Anti-Fragility Analysis Page...")

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

def plot_scenario_page(prices: pd.DataFrame, ticker: str = "SPY") -> plt.Figure:
    """Page 17: Scenario Analysis & Contingency Planning."""
    logger.info("Generating Scenario Analysis Page...")

    if ticker not in prices.columns: return None

    series = prices[ticker].dropna()
    current_price = series.iloc[-1]

    # 1. Calculate Probabilities
    # Using 3 months (60 days) horizon for standard table
    probs_1m = calculate_move_probabilities(series, days=21, n_sims=2000)
    probs_3m = calculate_move_probabilities(series, days=63, n_sims=2000)

    # 2. Generate Contingencies
    # Simple proxies for Regime classification (expand later)
    # Using 1M Vol approx
    curr_vol = series.pct_change().std() * np.sqrt(252)
    # Skew
    curr_skew = series.pct_change().dropna().skew()

    regime = "Normal"
    if curr_vol > 0.20: regime = "High Vol"
    elif curr_vol < 0.10: regime = "Low Vol"

    # Vol Score (0-1 normalized roughly)
    vol_score = min(max((curr_vol - 0.10) / 0.20, 0), 1)

    contingencies = generate_contingencies(current_price, regime, vol_score, curr_skew)

    # Plot
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("QUANT LAB: SCENARIO ANALYSIS & CONTINGENCIES", fontsize=16, weight='bold', y=0.98)

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

    # Panel 1: Probability Table (Text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title("1. Move Probabilities (Monte Carlo)", fontsize=12, weight='bold')

    # Build Table Text
    table_text = [["Target", "1-Month Prob", "3-Month Prob"]]
    for target in probs_1m.index:
        p1 = probs_1m.loc[target, "Upside Prob"]
        # p1_down = probs_1m.loc[target, "Downside Risk"]
        p3 = probs_3m.loc[target, "Upside Prob"]
        # p3_down = probs_3m.loc[target, "Downside Risk"]

        # Display as range? Or just Upside for now simpler
        # Actually show Upside vs Downside side by side?
        # Let's simplify: Display Probability of touching +/- X%

        row = [f"{target}", f"{p1:.1%}", f"{p3:.1%}"]
        table_text.append(row)

    table = ax1.table(cellText=table_text, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Panel 2: Visual Cone (Simplified from Page 12, focused on Targets)
    ax2 = fig.add_subplot(gs[0, 1])

    # Re-simulate for plot small batch
    mu = series.pct_change().mean() * 252
    sigma = series.pct_change().std() * np.sqrt(252)
    T = 63/252
    t_sim, paths = simulate_gbm(current_price, mu, sigma, T, 1/252, 100)

    future_dates = pd.date_range(start=series.index[-1], periods=len(t_sim), freq='B')

    # Plot Cone
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    p50 = np.percentile(paths, 50, axis=0)

    ax2.plot(future_dates, p50, 'b-', label="Median")
    ax2.fill_between(future_dates, p5, p95, color='blue', alpha=0.1, label="90% Cone")

    # Horizontal Lines for Targets
    for target_pct in [0.05, -0.05]:
        level = current_price * (1 + target_pct)
        color = 'green' if target_pct > 0 else 'red'
        ax2.axhline(level, linestyle='--', color=color, alpha=0.5)
        ax2.text(future_dates[0], level, f"{target_pct:+.0%} Target", color=color, fontsize=8, va='bottom')

    ax2.set_title("2. Target Visualization (3-Month)", fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Contingency Playbook (Bottom Full Width)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    ax3.set_title(f"3. TACTICAL CONTINGENCY PLAYBOOK (Regime: {regime})", fontsize=12, weight='bold')

    # Format DataFrame as Table
    cols = list(contingencies.columns)
    cell_text = []
    for row in contingencies.itertuples(index=False):
        cell_text.append(list(row))

    # Add colors based on Scenario type?
    colors = []
    # Create colors matrix matching dimensions (n_rows x n_cols)
    for row in contingencies["Scenario"]:
        row_colors = []
        if "Critical" in row or "Crash" in row: base_color = "#ffcccc" # Red tint
        elif "Euphoria" in row: base_color = "#ccffcc" # Green tint
        else: base_color = "white"

        for _ in range(len(cols)): row_colors.append(base_color)
        colors.append(row_colors)

    table3 = ax3.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='left',
                       colWidths=[0.25, 0.25, 0.35, 0.15],
                       cellColours=colors if colors else None)

    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2.0) # More vertical space

    # Footer
    text_content = (
        "INTERPRETATION GUIDE:\n"
        "1. MOVE PROBABILITIES: Probability of touching price levels based on current volatility regime.\n"
        "2. CONTINGENCIES: Pre-planned actions to remove emotion. If 'Condition' is met, execute 'Action'.\n"
        "3. FRAGILITY CHECK: Playbook adapts to Skew/Kurtosis. Negative Skew = Expensive Puts = Risk Reversals preferred."
    )
    fig.text(0.05, 0.02, text_content, fontsize=9, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkblue', linewidth=1.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    return fig
