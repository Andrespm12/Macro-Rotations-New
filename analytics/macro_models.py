"""
Core Macro Analytics: Regimes, Scoring, and Plumbing.
"""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, percentileofscore
from typing import Dict
from analytics.rotations import calculate_seasonality # Need imports
# from analytics.quant import calculate_recession_prob as calc_rec_ignored # REMOVED
from core.logger import get_logger

logger = get_logger(__name__)

# Helper
def calculate_recession_prob(spread_series: pd.Series) -> pd.Series:
    """Calculates Recession Probability using Probit Model."""
    intercept = -0.5333
    beta = -0.6330
    z = intercept + beta * spread_series
    prob = norm.cdf(z) * 100
    return pd.Series(prob, index=spread_series.index)

def calculate_regime_gmm(prices: pd.DataFrame) -> Dict:
    """Identifies Market Regime using GMM."""
    logger.info("Running GMM Regime Detection...")
    res = {"current_state": "N/A", "probs": [], "trans_matrix": None, "labels": []}
    
    if "SPY" not in prices.columns: return res
    
    spy = prices["SPY"]
    spy_w = spy.resample('W').last()
    rets = spy_w.pct_change().infer_objects(copy=False).dropna()
    vol = rets.rolling(4).std().dropna()
    
    df_feat = pd.DataFrame({"Ret": rets, "Vol": vol}).dropna()
    X = df_feat.values
    
    if len(X) < 52: return res
    
    try:
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)[-1]
        
        means = gmm.means_
        sorted_idx = np.argsort(means[:, 0])[::-1]
        
        state_map = {
            sorted_idx[0]: "BULL (Low Vol, Up)",
            sorted_idx[1]: "CHOPPY (Mixed)",
            sorted_idx[2]: "BEAR (High Vol, Down)"
        }
        
        current_label = labels[-1]
        current_state = state_map.get(current_label, "Unknown")
        
        trans_mat = np.zeros((3, 3))
        for (i, j) in zip(labels[:-1], labels[1:]):
            trans_mat[i][j] += 1
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
        next_probs = trans_mat[current_label]
        
        next_state_probs = {}
        for i in range(3):
            next_state_probs[state_map[i]] = next_probs[i]
            
        res = {
            "current_state": current_state,
            "probs": next_state_probs,
            "labels": labels,
            "dates": df_feat.index
        }
    except Exception as e:
        logger.error("GMM Error: %s", e)
        
    return res

def calculate_global_flows(prices: pd.DataFrame, macro: pd.DataFrame) -> Dict:
    """Calculates Global Capital Flow indicators."""
    logger.info("Calculating Global Capital Flows...")
    res = {
        "usd_strength": {"value": None, "status": "N/A"},
        "japan_carry": {"value": None, "status": "N/A"},
        "fed_custody": {"value": None, "status": "N/A"}
    }

    if "UUP" in prices.columns:
        uup = prices["UUP"].dropna()
        if len(uup) > 200:
            uup_ma200 = uup.rolling(200).mean().iloc[-1]
            curr_uup = uup.iloc[-1]
            status = "STRONG (Above MA)" if curr_uup > uup_ma200 else "WEAK (Below MA)"
            res["usd_strength"] = {"value": curr_uup, "status": status}

    if "JPY=X" in prices.columns and "10Y_Yield" in macro.columns:
        jpy = prices["JPY=X"].dropna()
        us_10y = macro["10Y_Yield"].dropna()
        common_idx = jpy.index.intersection(us_10y.index)
        jpy_aligned = jpy.loc[common_idx]
        us_10y_aligned = us_10y.loc[common_idx]

        if len(jpy_aligned) > 20:
            jpy_20d = (jpy_aligned.iloc[-1] / jpy_aligned.iloc[-20]) - 1
            us_10y_20d = (us_10y_aligned.iloc[-1] / us_10y_aligned.iloc[-20]) - 1

            if jpy_20d > 0.01 and us_10y_20d > 0.05:
                res["japan_carry"] = {"value": jpy_20d, "status": "UNWINDING (Stress)"}
            elif jpy_20d < -0.01 and us_10y_20d < -0.05:
                res["japan_carry"] = {"value": jpy_20d, "status": "ACCELERATING (Risk-On)"}
            else:
                res["japan_carry"] = {"value": jpy_20d, "status": "NEUTRAL"}
    
    if "Fed_Custody" in macro.columns:
        fed = macro["Fed_Custody"].dropna()
        if len(fed) > 52:
            yoy = (fed.iloc[-1] / fed.iloc[-52]) - 1
            if yoy > 0.02: status = "INCREASING (Demand)"
            elif yoy < -0.02: status = "DECREASING (Withdrawal)"
            else: status = "STABLE"
            res["fed_custody"] = {"value": yoy, "status": status}
    return res

def calculate_liquidity_stress(df: pd.DataFrame) -> Dict:
    """Calculates Liquidity Stress (Credit Spread + VIX)."""
    logger.info("Calculating Liquidity Stress...")
    res = {"level": None, "status": "N/A"}
    
    if "Spread_Credit" in df.columns and "CF_VIX" in df.columns:
        spread = df["Spread_Credit"].iloc[-1]
        vix = df["CF_VIX"].iloc[-1]
        
        spread_mean = df["Spread_Credit"].rolling(252).mean().iloc[-1]
        spread_std = df["Spread_Credit"].rolling(252).std().iloc[-1]
        spread_z = (spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        vix_mean = df["CF_VIX"].rolling(252).mean().iloc[-1]
        vix_std = df["CF_VIX"].rolling(252).std().iloc[-1]
        vix_z = (vix - vix_mean) / vix_std if vix_std > 0 else 0
        
        stress_score = (spread_z + vix_z) / 2
        
        if stress_score > 2.0: status = "CRITICAL (Liquidity Crunch)"
        elif stress_score > 1.0: status = "ELEVATED (Tightening)"
        else: status = "NORMAL (Abundant)"
        
        res = {"level": stress_score, "status": status, "spread_z": spread_z, "vix_z": vix_z}
    return res

def build_analytics(prices: pd.DataFrame, macro: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Constructs ALL ratios: The Original Rotations + The Capital Flows Plumbing."""
    # Build all columns in a dict first, then concat once to avoid fragmentation
    cols = {}

    def ratio(n, d):
        return prices[n] / prices[d]

    # PART A: Capital Flows
    cols["CF_Liquidity_Valve"] = ratio("BTC-USD", "GLD")
    cols["CF_Real_Yield"] = macro["10Y_Yield"] - macro["10Y_Breakeven"]
    cols["CF_Breadth"] = ratio("RSP", "SPY")
    cols["CF_VIX"] = macro["VIX"]
    cols["CF_VIX3M"] = macro["VIX3M"] if "VIX3M" in macro.columns else pd.Series(np.nan, index=prices.index)
    cols["CF_NFCI"] = macro["Fin_Conditions"]

    # Raw Macro pass-through
    macro_passthrough = [
        "10Y_Yield", "2Y_Yield", "HY_Spread",
        "5Y_Breakeven", "10Y_Breakeven", "5Y5Y_Forward",
        "SOFR", "Fed_Funds",
        "M2_YoY", "Fed_Assets_YoY", "M2_Velocity", "M2_Velocity_YoY",
    ]
    for col_name in macro_passthrough:
        if col_name in macro.columns:
            cols[col_name] = macro[col_name]

    # Credit Spread proxy
    if "HY_Spread" in macro.columns:
        cols["Spread_Credit"] = macro["HY_Spread"]

    # Price pass-through
    if "DX-Y.NYB" in prices.columns:
        cols["DX-Y.NYB"] = prices["DX-Y.NYB"]
    if "^MOVE" in prices.columns:
        cols["MOVE_Index"] = prices["^MOVE"]

    # CPI YoY
    if "CPI_YoY" in macro.columns:
        cols["CPI_YoY"] = macro["CPI_YoY"]
    elif "CPI" in macro.columns:
        cols["CPI_YoY"] = macro["CPI"].pct_change(252).infer_objects(copy=False)

    if "Unemployment" in macro.columns:
        cols["Unemployment"] = macro["Unemployment"] / 100

    cols["CF_Consumer"] = ratio("XLY", "XLP")
    
    # PART B: Rotations
    cols["ROT_Value_Growth"] = ratio("RPV", "VONG")
    cols["ROT_HiBeta_LoVol"] = ratio("SPHB", "SPLV")
    cols["ROT_Small_Large"] = ratio("IWM", "QQQ")
    cols["ROT_Quality_Mkt"] = ratio("NOBL", "SPY")
    cols["ROT_Biotech_Mkt"] = ratio("XBI", "SPY")

    cyclicals = prices["XLI"] + prices["XLB"]
    defensives = prices["XLU"] + prices["XLP"]
    cols["ROT_Cyc_Def_Sectors"] = cyclicals / defensives

    labor = (prices["MAN"] + prices["RHI"] + prices["KELYA"]) / 3
    cols["ROT_Labor_SPY"] = labor / prices["SPY"]

    cols["ROT_US_World"] = ratio("SPY", "ACWX")
    cols["ROT_EM_DM"] = ratio("EEM", "VEA")
    cols["ROT_Copper_Gold"] = ratio("CPER", "GLD")
    cols["ROT_Fin_Tech"] = ratio("XLF", "XLK")

    if "SPY" in prices.columns and "TLT" in prices.columns:
        spy_ret = prices["SPY"].pct_change().infer_objects(copy=False)
        tlt_ret = prices["TLT"].pct_change().infer_objects(copy=False)
        cols["ROT_SPY_TLT_Corr"] = spy_ret.rolling(60).corr(tlt_ret)
    else:
        cols["ROT_SPY_TLT_Corr"] = pd.Series(0.0, index=prices.index)

    # PART C: Rate Diffs
    if "Germany_10Y" in macro.columns:
        cols["RateDiff_US_DE"] = macro["10Y_Yield"] - macro["Germany_10Y"]
    if "Japan_10Y" in macro.columns:
        cols["RateDiff_US_JP"] = macro["10Y_Yield"] - macro["Japan_10Y"]
    if "UK_10Y" in macro.columns:
        cols["RateDiff_US_UK"] = macro["10Y_Yield"] - macro["UK_10Y"]

    # PART D: Models
    if "Spread_10Y3M" in macro.columns:
        cols["Recession_Prob_Model"] = calculate_recession_prob(macro["Spread_10Y3M"])
        cols["Spread_10Y3M"] = macro["Spread_10Y3M"]
        cols["Recession_Prob"] = cols["Recession_Prob_Model"]
    elif "10Y_Yield" in macro.columns and "3M_Yield" in macro.columns:
        spread_10y3m = macro["10Y_Yield"] - macro["3M_Yield"]
        cols["Recession_Prob_Model"] = calculate_recession_prob(spread_10y3m)
        cols["Spread_10Y3M"] = spread_10y3m
        cols["Recession_Prob"] = cols["Recession_Prob_Model"]

    if "Recession_Prob" in macro.columns:
        cols["FRED_Recession_Prob"] = macro["Recession_Prob"]
        if not macro["Recession_Prob"].dropna().empty:
            cols["Recession_Prob"] = macro["Recession_Prob"]

    # Build DataFrame in one operation to avoid fragmentation
    df = pd.DataFrame(cols, index=prices.index)

    # PART E: Z-Scores (batch computation to avoid DataFrame fragmentation)
    window = 252
    z_score_cols = {}
    for col in df.columns:
        if "MA200" in col:
            continue
        roll_mean = df[col].rolling(window).mean()
        roll_std = df[col].rolling(window).std()
        z_score_cols[f"{col}_Z"] = (df[col] - roll_mean) / roll_std

    if z_score_cols:
        df = pd.concat([df, pd.DataFrame(z_score_cols, index=df.index)], axis=1)

    return df

def add_trends_and_score(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Adds MAs and calculates the composite Macro Score."""
    new_cols = {}
    for col in df.columns:
        # Skip if already exists or isn't numeric (optional check, but rolling works on numeric)
        if pd.api.types.is_numeric_dtype(df[col]):
            new_cols[f"{col}_MA200"] = df[col].rolling(config["ma_long"]).mean()
            
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
    risk_map = {
        "CF_Liquidity_Valve": 1,
        "CF_Consumer": 1,
        "CF_Breadth": 1,
        "ROT_Value_Growth": 1,
        "ROT_HiBeta_LoVol": 1,
        "ROT_Small_Large": 1,
        "ROT_Biotech_Mkt": 1,
        "ROT_Cyc_Def_Sectors": 1,
        "ROT_Labor_SPY": 1,
        "ROT_US_World": 1,
        "ROT_EM_DM": 1,
        "ROT_Copper_Gold": 1,
        "ROT_Fin_Tech": 1,
        "ROT_Quality_Mkt": -1 
    }
    
    score_series = pd.Series(0.0, index=df.index)
    valid_count = 0
    threshold = config["regime_threshold"]
    
    for col, direction in risk_map.items():
        if col not in df.columns: continue
        ma_col = f"{col}_MA200"
        if ma_col not in df.columns: continue
        
        ma = df[ma_col]
        # Avoid division by zero
        dist = (df[col] / ma - 1.0) * 100
        
        raw_score = pd.Series(0.0, index=df.index)
        # Handle NAs in dist
        dist = dist.fillna(0)
        
        raw_score[dist > threshold] = 1.0
        raw_score[dist < -threshold] = -1.0
        
        score_series += (raw_score * direction)
        valid_count += 1
        
    if valid_count > 0:
        df["MACRO_SCORE"] = score_series / valid_count

    return df.copy() # Return copy to de-fragment

def calculate_macro_radar(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates the 1-Year Percentile Rank (0-100) for the Macro Radar."""
    window = 252
    subset = df.iloc[-window:].copy()
    factors = {}
    
    if "CF_Consumer" in subset.columns: factors["Growth"] = subset["CF_Consumer"]
    if "M2_YoY" in subset.columns: factors["Liquidity"] = subset["M2_YoY"]
    if "CPI_YoY" in subset.columns: factors["Inflation"] = subset["CPI_YoY"]
    if "TLT" in prices.columns: factors["Rates"] = prices["TLT"].iloc[-window:]
    if "CF_Liquidity_Valve" in subset.columns: factors["Risk"] = subset["CF_Liquidity_Valve"]
    if "CF_VIX" in subset.columns: factors["Sentiment"] = -subset["CF_VIX"]
        
    ranks = {}
    current_values = {}
    
    for name, series in factors.items():
        curr = series.iloc[-1]
        rank = percentileofscore(series.dropna(), curr)
        ranks[name] = rank
        current_values[name] = curr
        
    return pd.DataFrame({"Rank": ranks, "Value": current_values})
