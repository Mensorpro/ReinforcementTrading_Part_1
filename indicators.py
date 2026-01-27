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

    # Columns the AGENT should see (scale-invariant features only)
    # Reduced from 29 to 15 features to combat overfitting
    feature_cols = [
        # Momentum (3 features - removed redundant RSI_9, MACD_norm, stoch_d, roc_10)
        "rsi_14",           # Overbought/oversold (slower, more reliable)
        "macd_hist_norm",   # Momentum divergence (histogram captures trend changes)
        "stoch_k",          # Fast oscillator (complements RSI)
        
        # Volatility (3 features - keep all, non-redundant)
        "bb_width",         # Volatility expansion/contraction
        "bb_position",      # Price position in bands (mean reversion signal)
        "atr_change",       # Volatility regime shifts
        
        # Trend (6 features - removed redundant EMA distances/slopes/spreads)
        "adx",              # Trend strength (key for filtering)
        "di_plus",          # Bullish directional pressure
        "di_minus",         # Bearish directional pressure
        "close_ema21_dist", # Medium-term trend distance (most useful timeframe)
        "ema_21_slope",     # Medium-term trend direction
        "ema_50_200_spread",# Long-term trend (golden/death cross)
        
        # Volume (2 features - keep all, non-redundant)
        "obv_momentum",     # Accumulation/distribution
        "volume_roc",       # Volume spikes (breakout confirmation)
        
        # Candle patterns (1 feature - removed redundant wick/range features)
        "candle_body",      # Bullish/bearish pressure (most informative)
    ]

    return df, feature_cols
