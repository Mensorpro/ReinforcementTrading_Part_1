import numpy as np
import pandas as pd
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns: [Time (EET), Open, High, Low, Close, Volume]
    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    df = pd.read_csv(csv_path)
    
    # Strip any trailing spaces in headers (e.g. 'Volume ')
    df.columns = df.columns.str.strip()
    
    # Detect the time column (could be "Gmt time", "Time (EET)", etc.)
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No time/date column found in CSV. Columns: {df.columns.tolist()}")
    
    # Parse datetime and set as index
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
    df = df.set_index(time_col)
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Technicals ----
    
    # === MOMENTUM INDICATORS ===
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["rsi_9"] = ta.rsi(df["Close"], length=9)  # Faster RSI
    
    # MACD (trend following momentum)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    
    # Stochastic (overbought/oversold)
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    
    # Rate of Change (momentum)
    df["roc_10"] = ta.roc(df["Close"], length=10)
    
    # === VOLATILITY INDICATORS ===
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_7"] = ta.atr(df["High"], df["Low"], df["Close"], length=7)  # Faster ATR
    
    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=20, std=2)
    # Column names vary by pandas_ta version, try both formats
    bb_upper_col = [c for c in bbands.columns if 'BBU' in c][0]
    bb_middle_col = [c for c in bbands.columns if 'BBM' in c][0]
    bb_lower_col = [c for c in bbands.columns if 'BBL' in c][0]
    
    df["bb_upper"] = bbands[bb_upper_col]
    df["bb_middle"] = bbands[bb_middle_col]
    df["bb_lower"] = bbands[bb_lower_col]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]  # Normalized width
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # Where price is in band
    
    # === TREND INDICATORS ===
    # EMAs (more responsive than SMA)
    df["ema_9"] = ta.ema(df["Close"], length=9)
    df["ema_21"] = ta.ema(df["Close"], length=21)
    df["ema_50"] = ta.ema(df["Close"], length=50)
    df["ema_200"] = ta.ema(df["Close"], length=200)
    
    # ADX (trend strength)
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx"] = adx["ADX_14"]
    df["di_plus"] = adx["DMP_14"]
    df["di_minus"] = adx["DMN_14"]
    
    # === VOLUME INDICATORS ===
    # On-Balance Volume (accumulation/distribution)
    df["obv"] = ta.obv(df["Close"], df["Volume"])
    df["obv_ema"] = ta.ema(df["obv"], length=20)
    
    # Volume Rate of Change
    df["volume_roc"] = ta.roc(df["Volume"], length=10)
    
    # === DERIVED FEATURES (Scale-Invariant) ===
    
    # EMA distances (normalized by ATR)
    df["close_ema9_dist"] = (df["Close"] - df["ema_9"]) / df["atr_14"]
    df["close_ema21_dist"] = (df["Close"] - df["ema_21"]) / df["atr_14"]
    df["close_ema50_dist"] = (df["Close"] - df["ema_50"]) / df["atr_14"]
    df["close_ema200_dist"] = (df["Close"] - df["ema_200"]) / df["atr_14"]
    
    # EMA slopes (normalized)
    df["ema_9_slope"] = df["ema_9"].diff() / df["atr_14"]
    df["ema_21_slope"] = df["ema_21"].diff() / df["atr_14"]
    df["ema_50_slope"] = df["ema_50"].diff() / df["atr_14"]
    
    # EMA crossover signals
    df["ema_9_21_spread"] = (df["ema_9"] - df["ema_21"]) / df["atr_14"]
    df["ema_21_50_spread"] = (df["ema_21"] - df["ema_50"]) / df["atr_14"]
    df["ema_50_200_spread"] = (df["ema_50"] - df["ema_200"]) / df["atr_14"]
    
    # MACD normalized
    df["macd_norm"] = df["macd"] / df["atr_14"]
    df["macd_hist_norm"] = df["macd_hist"] / df["atr_14"]
    
    # ATR change (volatility regime)
    df["atr_change"] = df["atr_14"].pct_change()
    
    # OBV momentum
    df["obv_momentum"] = (df["obv"] - df["obv_ema"]) / df["obv_ema"].abs()
    
    # Candle patterns (relative to ATR)
    df["candle_body"] = (df["Close"] - df["Open"]) / df["atr_14"]
    df["candle_range"] = (df["High"] - df["Low"]) / df["atr_14"]
    df["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["atr_14"]
    df["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["atr_14"]
    
    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # === PROPER NORMALIZATION (User-specified formulas) ===
    
    # 1. RSI: Normalize to [-1, 1]: RSI/50 - 1
    df["rsi_14_norm"] = df["rsi_14"] / 50.0 - 1.0
    df["stoch_k_norm"] = df["stoch_k"] / 50.0 - 1.0
    
    # 2. MACD: Use tanh scaling for bounded output
    df["macd_hist_tanh"] = np.tanh(df["macd_hist_norm"] / 0.05)  # s=0.05 for MACD histogram
    
    # 3. ATR: Scale relative to price, then normalize
    df["atr_relative"] = df["atr_14"] / df["Close"]
    # Rescale to [-1, 1] using typical range (0.001 to 0.01 for EURUSD)
    df["atr_norm"] = np.clip((df["atr_relative"] - 0.005) / 0.005, -1.0, 1.0)
    
    # 4. BB width: Already relative, normalize to [-1, 1] (typical range 0.01-0.05)
    df["bb_width_norm"] = np.clip((df["bb_width"] - 0.03) / 0.03, -1.0, 1.0)
    
    # 5. BB position: Already in [0, 1], convert to [-1, 1] and clip outliers
    df["bb_position_norm"] = np.clip(df["bb_position"] * 2.0 - 1.0, -1.0, 1.0)
    
    # 6. ATR change: Clip to reasonable range
    df["atr_change_norm"] = np.clip(df["atr_change"] / 0.1, -1.0, 1.0)
    
    # 7. ADX: Normalize to [0, 1] (ADX is 0-100)
    df["adx_norm"] = df["adx"] / 100.0
    
    # 8. DI+/DI-: Normalize to [0, 1] (typically 0-50)
    df["di_plus_norm"] = np.clip(df["di_plus"] / 50.0, 0.0, 1.0)
    df["di_minus_norm"] = np.clip(df["di_minus"] / 50.0, 0.0, 1.0)
    
    # 9. EMA distances: Already normalized by ATR, just clip
    df["close_ema21_dist_norm"] = np.clip(df["close_ema21_dist"] / 5.0, -1.0, 1.0)
    df["ema_21_slope_norm"] = np.clip(df["ema_21_slope"] / 2.0, -1.0, 1.0)
    df["ema_50_200_spread_norm"] = np.clip(df["ema_50_200_spread"] / 10.0, -1.0, 1.0)
    
    # 10. OBV momentum: Clip to reasonable range
    df["obv_momentum_norm"] = np.clip(df["obv_momentum"] / 0.5, -1.0, 1.0)
    
    # 11. Volume ROC: Clip to prevent explosions (seen in live data)
    df["volume_roc_norm"] = np.clip(df["volume_roc"] / 100.0, -1.0, 1.0)
    
    # 12. Candle body: Already normalized by ATR, just clip
    df["candle_body_norm"] = np.clip(df["candle_body"] / 2.0, -1.0, 1.0)
    
    # === TEMPORAL CONTEXT (Cyclical Time Encoding) ===
    # Sin/Cos encoding for hour of day (captures session liquidity patterns)
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week encoding (captures weekly patterns)
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Drop any remaining NaNs from normalization
    df.dropna(inplace=True)

    # Columns the AGENT should see (NORMALIZED features in [-1, 1] range)
    # Following "The Three Eyes" professional strategy:
    # A. Market Context (15 features)
    # B. Account State (3 features - added in environment)
    # C. Temporal Context (4 features - cyclical time)
    feature_cols = [
        # === A. MARKET CONTEXT (15 features) ===
        
        # Momentum (3 features)
        "rsi_14_norm",           # Overbought/oversold [-1, 1]
        "macd_hist_tanh",        # Momentum divergence [-1, 1]
        "stoch_k_norm",          # Fast oscillator [-1, 1]
        
        # Volatility (3 features)
        "bb_width_norm",         # Volatility expansion/contraction [-1, 1]
        "bb_position_norm",      # Price position in bands [-1, 1]
        "atr_change_norm",       # Volatility regime shifts [-1, 1]
        
        # Trend (6 features)
        "adx_norm",              # Trend strength [0, 1]
        "di_plus_norm",          # Bullish directional pressure [0, 1]
        "di_minus_norm",         # Bearish directional pressure [0, 1]
        "close_ema21_dist_norm", # Medium-term trend distance [-1, 1]
        "ema_21_slope_norm",     # Medium-term trend direction [-1, 1]
        "ema_50_200_spread_norm",# Long-term trend [-1, 1]
        
        # Volume (2 features)
        "obv_momentum_norm",     # Accumulation/distribution [-1, 1]
        "volume_roc_norm",       # Volume spikes [-1, 1]
        
        # Candle patterns (1 feature)
        "candle_body_norm",      # Bullish/bearish pressure [-1, 1]
        
        # === C. TEMPORAL CONTEXT (4 features) ===
        "hour_sin",              # Hour of day (cyclical) [-1, 1]
        "hour_cos",              # Hour of day (cyclical) [-1, 1]
        "day_sin",               # Day of week (cyclical) [-1, 1]
        "day_cos",               # Day of week (cyclical) [-1, 1]
    ]

    return df, feature_cols
