import numpy as np
import pandas as pd
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and builds a FLAT feature vector per bar.

    The key insight: instead of passing a 2D (window x features) matrix to MLP
    (which destroys temporal structure), we precompute temporal features
    (momentum, regime, price-action) so each bar is a self-contained 1D vector
    that the MLP can actually learn meaningful patterns from.

    Returns:
        df: DataFrame with OHLCV + all computed columns
        feature_cols: list of column names the agent should see (1D per bar)
    """
    df = pd.read_csv(csv_path)

    # Strip trailing spaces in headers
    df.columns = df.columns.str.strip()

    # Detect time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_col = col
            break
    if time_col is None:
        raise ValueError(f"No time/date column found. Columns: {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
    df = df.set_index(time_col)
    df.sort_index(inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ================================================================
    # RAW INDICATORS
    # ================================================================

    # Momentum
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]

    # Volatility
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    bbands = ta.bbands(df["Close"], length=20, std=2)
    bb_upper_col = [c for c in bbands.columns if 'BBU' in c][0]
    bb_middle_col = [c for c in bbands.columns if 'BBM' in c][0]
    bb_lower_col = [c for c in bbands.columns if 'BBL' in c][0]
    df["bb_upper"] = bbands[bb_upper_col]
    df["bb_middle"] = bbands[bb_middle_col]
    df["bb_lower"] = bbands[bb_lower_col]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Trend
    df["ema_9"] = ta.ema(df["Close"], length=9)
    df["ema_21"] = ta.ema(df["Close"], length=21)
    df["ema_50"] = ta.ema(df["Close"], length=50)
    df["ema_200"] = ta.ema(df["Close"], length=200)
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx"] = adx["ADX_14"]
    df["di_plus"] = adx["DMP_14"]
    df["di_minus"] = adx["DMN_14"]

    # Volume
    df["obv"] = ta.obv(df["Close"], df["Volume"])
    df["obv_ema"] = ta.ema(df["obv"], length=20)
    df["volume_roc"] = ta.roc(df["Volume"], length=10)

    # Derived (ATR-normalized, scale-invariant)
    df["close_ema21_dist"] = (df["Close"] - df["ema_21"]) / df["atr_14"]
    df["ema_21_slope"] = df["ema_21"].diff() / df["atr_14"]
    df["ema_50_200_spread"] = (df["ema_50"] - df["ema_200"]) / df["atr_14"]
    df["macd_hist_norm"] = df["macd_hist"] / df["atr_14"]
    df["atr_change"] = df["atr_14"].pct_change()
    df["obv_momentum"] = (df["obv"] - df["obv_ema"]) / df["obv_ema"].abs().clip(lower=1)
    df["candle_body"] = (df["Close"] - df["Open"]) / df["atr_14"]

    # Drop initial NaNs from indicator warmup
    df.dropna(inplace=True)

    # ================================================================
    # NORMALIZED FEATURES (all roughly in [-1, 1])
    # ================================================================

    # --- A. Current Market State (15 features) ---
    df["rsi_14_norm"] = df["rsi_14"] / 50.0 - 1.0
    df["stoch_k_norm"] = df["stoch_k"] / 50.0 - 1.0
    df["macd_hist_tanh"] = np.tanh(df["macd_hist_norm"] / 0.05)
    df["bb_width_norm"] = np.clip((df["bb_width"] - 0.03) / 0.03, -1.0, 1.0)
    df["bb_position_norm"] = np.clip(df["bb_position"] * 2.0 - 1.0, -1.0, 1.0)
    df["atr_change_norm"] = np.clip(df["atr_change"] / 0.1, -1.0, 1.0)
    df["adx_norm"] = df["adx"] / 100.0
    df["di_plus_norm"] = np.clip(df["di_plus"] / 50.0, 0.0, 1.0)
    df["di_minus_norm"] = np.clip(df["di_minus"] / 50.0, 0.0, 1.0)
    df["close_ema21_dist_norm"] = np.clip(df["close_ema21_dist"] / 5.0, -1.0, 1.0)
    df["ema_21_slope_norm"] = np.clip(df["ema_21_slope"] / 2.0, -1.0, 1.0)
    df["ema_50_200_spread_norm"] = np.clip(df["ema_50_200_spread"] / 10.0, -1.0, 1.0)
    df["obv_momentum_norm"] = np.clip(df["obv_momentum"] / 0.5, -1.0, 1.0)
    df["volume_roc_norm"] = np.clip(df["volume_roc"] / 100.0, -1.0, 1.0)
    df["candle_body_norm"] = np.clip(df["candle_body"] / 2.0, -1.0, 1.0)

    # --- B. Temporal Context (4 features) ---
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ================================================================
    # TEMPORAL FEATURES  (encode recent history as flat features)
    # These let an MLP understand temporal dynamics without a window.
    # ================================================================

    # --- C. Multi-bar Momentum (10 features) ---
    # How much key features changed over 5 and 15 bars
    _momentum_sources = [
        "rsi_14_norm", "macd_hist_tanh", "bb_position_norm",
        "close_ema21_dist_norm", "adx_norm",
    ]
    for feat in _momentum_sources:
        df[f"{feat}_chg5"]  = np.clip(df[feat].diff(5),  -1.0, 1.0)
        df[f"{feat}_chg15"] = np.clip(df[feat].diff(15), -1.0, 1.0)

    # --- D. Regime Detection (5 features) ---
    # Rolling statistics capture "what state the market has been in"
    df["rsi_regime"]     = df["rsi_14_norm"].rolling(20).mean()
    df["vol_regime"]     = df["bb_width_norm"].rolling(20).mean()
    df["trend_regime"]   = df["adx_norm"].rolling(20).mean()
    df["body_bias"]      = df["candle_body_norm"].rolling(10).mean()
    df["vol_spikiness"]  = np.clip(df["volume_roc_norm"].rolling(10).std(), 0, 1)

    # --- E. Price Action Patterns (4 features) ---
    # Short and medium term price momentum (ATR-normalized)
    atr_relative = df["atr_14"] / df["Close"]  # fractional ATR
    df["price_mom_5"]  = np.clip(df["Close"].pct_change(5)  / atr_relative, -2, 2) / 2.0
    df["price_mom_20"] = np.clip(df["Close"].pct_change(20) / atr_relative, -2, 2) / 2.0

    # Range expansion: how wide was recent price range vs typical?
    df["range_expansion_20"] = np.clip(
        (df["High"].rolling(20).max() - df["Low"].rolling(20).min())
        / (df["atr_14"] * 20), 0, 1
    )

    # Trend consistency: ratio of higher-highs minus lower-lows over 10 bars
    df["_hh"] = (df["High"] > df["High"].shift(1)).astype(float)
    df["_ll"] = (df["Low"]  < df["Low"].shift(1)).astype(float)
    df["trend_consistency"] = (
        df["_hh"].rolling(10).mean() - df["_ll"].rolling(10).mean()
    )  # in [-1, 1]

    # Clean up temp columns
    df.drop(columns=["_hh", "_ll"], inplace=True)

    # Drop NaNs introduced by temporal features
    df.dropna(inplace=True)

    # ================================================================
    # FEATURE LIST  (38 features — 1D vector per bar for MLP)
    # ================================================================
    feature_cols = [
        # A. Current Market State (15)
        "rsi_14_norm",
        "macd_hist_tanh",
        "stoch_k_norm",
        "bb_width_norm",
        "bb_position_norm",
        "atr_change_norm",
        "adx_norm",
        "di_plus_norm",
        "di_minus_norm",
        "close_ema21_dist_norm",
        "ema_21_slope_norm",
        "ema_50_200_spread_norm",
        "obv_momentum_norm",
        "volume_roc_norm",
        "candle_body_norm",

        # B. Temporal Context (4)
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",

        # C. Multi-bar Momentum (10)
        "rsi_14_norm_chg5",
        "rsi_14_norm_chg15",
        "macd_hist_tanh_chg5",
        "macd_hist_tanh_chg15",
        "bb_position_norm_chg5",
        "bb_position_norm_chg15",
        "close_ema21_dist_norm_chg5",
        "close_ema21_dist_norm_chg15",
        "adx_norm_chg5",
        "adx_norm_chg15",

        # D. Regime Detection (5)
        "rsi_regime",
        "vol_regime",
        "trend_regime",
        "body_bias",
        "vol_spikiness",

        # E. Price Action Patterns (4)
        "price_mom_5",
        "price_mom_20",
        "range_expansion_20",
        "trend_consistency",
    ]

    return df, feature_cols
