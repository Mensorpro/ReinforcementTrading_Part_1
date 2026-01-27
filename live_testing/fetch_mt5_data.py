"""
Fetch live EUR/USD data from MetaTrader 5 and prepare it for testing
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas_ta as ta


def fetch_eurusd_data(timeframe=mt5.TIMEFRAME_H1, bars=10000):
    """
    Fetch EUR/USD data from MT5
    
    Args:
        timeframe: MT5 timeframe constant (default: H1)
        bars: Number of bars to fetch (default: 10000 - maximum available)
    
    Returns:
        DataFrame with OHLC data
    """
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return None
    
    print("=" * 60)
    print("FETCHING DATA FROM MT5")
    print("=" * 60)
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Broker: {account_info.company}")
    
    symbol = "EURUSDm"
    
    # Check if symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        mt5.shutdown()
        return None
    
    print(f"\nSymbol: {symbol}")
    print(f"Timeframe: H1 (1 Hour)")
    print(f"Requesting: {bars:,} bars")
    
    # Fetch data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch data, error: {mt5.last_error()}")
        mt5.shutdown()
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Rename columns to match training data format
    df = df.rename(columns={
        'time': 'Time',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    })
    
    # Keep only OHLCV columns
    df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"\n✓ Fetched {len(df):,} bars")
    print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")
    print(f"Latest close: {df['Close'].iloc[-1]:.5f}")
    
    # Shutdown MT5
    mt5.shutdown()
    
    return df


def prepare_data_for_testing(df):
    """
    Add technical indicators to match training data format
    
    Args:
        df: Raw OHLC DataFrame with Time column
    
    Returns:
        DataFrame with indicators, list of feature columns
    """
    print("\n" + "=" * 60)
    print("ADDING TECHNICAL INDICATORS")
    print("=" * 60)
    
    # Set Time as index for indicator calculation
    df_indexed = df.set_index('Time')
    
    # Add all indicators (same as training data)
    # === MOMENTUM INDICATORS ===
    df_indexed["rsi_14"] = ta.rsi(df_indexed["Close"], length=14)
    df_indexed["rsi_9"] = ta.rsi(df_indexed["Close"], length=9)
    
    macd = ta.macd(df_indexed["Close"], fast=12, slow=26, signal=9)
    df_indexed["macd"] = macd["MACD_12_26_9"]
    df_indexed["macd_signal"] = macd["MACDs_12_26_9"]
    df_indexed["macd_hist"] = macd["MACDh_12_26_9"]
    
    stoch = ta.stoch(df_indexed["High"], df_indexed["Low"], df_indexed["Close"], k=14, d=3)
    df_indexed["stoch_k"] = stoch["STOCHk_14_3_3"]
    df_indexed["stoch_d"] = stoch["STOCHd_14_3_3"]
    
    df_indexed["roc_10"] = ta.roc(df_indexed["Close"], length=10)
    
    # === VOLATILITY INDICATORS ===
    df_indexed["atr_14"] = ta.atr(df_indexed["High"], df_indexed["Low"], df_indexed["Close"], length=14)
    df_indexed["atr_7"] = ta.atr(df_indexed["High"], df_indexed["Low"], df_indexed["Close"], length=7)
    
    bbands = ta.bbands(df_indexed["Close"], length=20, std=2)
    bb_upper_col = [c for c in bbands.columns if 'BBU' in c][0]
    bb_middle_col = [c for c in bbands.columns if 'BBM' in c][0]
    bb_lower_col = [c for c in bbands.columns if 'BBL' in c][0]
    
    df_indexed["bb_upper"] = bbands[bb_upper_col]
    df_indexed["bb_middle"] = bbands[bb_middle_col]
    df_indexed["bb_lower"] = bbands[bb_lower_col]
    df_indexed["bb_width"] = (df_indexed["bb_upper"] - df_indexed["bb_lower"]) / df_indexed["bb_middle"]
    df_indexed["bb_position"] = (df_indexed["Close"] - df_indexed["bb_lower"]) / (df_indexed["bb_upper"] - df_indexed["bb_lower"])
    
    # === TREND INDICATORS ===
    df_indexed["ema_9"] = ta.ema(df_indexed["Close"], length=9)
    df_indexed["ema_21"] = ta.ema(df_indexed["Close"], length=21)
    df_indexed["ema_50"] = ta.ema(df_indexed["Close"], length=50)
    df_indexed["ema_200"] = ta.ema(df_indexed["Close"], length=200)
    
    adx = ta.adx(df_indexed["High"], df_indexed["Low"], df_indexed["Close"], length=14)
    df_indexed["adx"] = adx["ADX_14"]
    df_indexed["di_plus"] = adx["DMP_14"]
    df_indexed["di_minus"] = adx["DMN_14"]
    
    # === VOLUME INDICATORS ===
    df_indexed["obv"] = ta.obv(df_indexed["Close"], df_indexed["Volume"])
    df_indexed["obv_ema"] = ta.ema(df_indexed["obv"], length=20)
    df_indexed["volume_roc"] = ta.roc(df_indexed["Volume"], length=10)
    
    # CRITICAL FIX: Clip volume_roc to match training data distribution
    # Training data 99.9th percentile is ~2.5e9, so clip there
    # This prevents MT5 tick volume from creating out-of-distribution values
    df_indexed["volume_roc"] = df_indexed["volume_roc"].clip(lower=-100, upper=2.5e9)
    
    # === DERIVED FEATURES ===
    df_indexed["close_ema9_dist"] = (df_indexed["Close"] - df_indexed["ema_9"]) / df_indexed["atr_14"]
    df_indexed["close_ema21_dist"] = (df_indexed["Close"] - df_indexed["ema_21"]) / df_indexed["atr_14"]
    df_indexed["close_ema50_dist"] = (df_indexed["Close"] - df_indexed["ema_50"]) / df_indexed["atr_14"]
    df_indexed["close_ema200_dist"] = (df_indexed["Close"] - df_indexed["ema_200"]) / df_indexed["atr_14"]
    
    df_indexed["ema_9_slope"] = df_indexed["ema_9"].diff() / df_indexed["atr_14"]
    df_indexed["ema_21_slope"] = df_indexed["ema_21"].diff() / df_indexed["atr_14"]
    df_indexed["ema_50_slope"] = df_indexed["ema_50"].diff() / df_indexed["atr_14"]
    
    df_indexed["ema_9_21_spread"] = (df_indexed["ema_9"] - df_indexed["ema_21"]) / df_indexed["atr_14"]
    df_indexed["ema_21_50_spread"] = (df_indexed["ema_21"] - df_indexed["ema_50"]) / df_indexed["atr_14"]
    df_indexed["ema_50_200_spread"] = (df_indexed["ema_50"] - df_indexed["ema_200"]) / df_indexed["atr_14"]
    
    df_indexed["macd_norm"] = df_indexed["macd"] / df_indexed["atr_14"]
    df_indexed["macd_hist_norm"] = df_indexed["macd_hist"] / df_indexed["atr_14"]
    
    df_indexed["atr_change"] = df_indexed["atr_14"].pct_change()
    
    # CRITICAL FIX: Clip atr_change to prevent extreme values
    df_indexed["atr_change"] = df_indexed["atr_change"].clip(lower=-1, upper=10)
    
    df_indexed["obv_momentum"] = (df_indexed["obv"] - df_indexed["obv_ema"]) / df_indexed["obv_ema"].abs()
    
    # CRITICAL FIX: Clip obv_momentum to prevent extreme values
    df_indexed["obv_momentum"] = df_indexed["obv_momentum"].clip(lower=-10, upper=10)
    
    df_indexed["candle_body"] = (df_indexed["Close"] - df_indexed["Open"]) / df_indexed["atr_14"]
    df_indexed["candle_range"] = (df_indexed["High"] - df_indexed["Low"]) / df_indexed["atr_14"]
    df_indexed["upper_wick"] = (df_indexed["High"] - df_indexed[["Open", "Close"]].max(axis=1)) / df_indexed["atr_14"]
    df_indexed["lower_wick"] = (df_indexed[["Open", "Close"]].min(axis=1) - df_indexed["Low"]) / df_indexed["atr_14"]
    
    # Feature columns (same as training)
    # Reduced from 29 to 15 to combat overfitting
    feature_cols = [
        # Momentum (3)
        "rsi_14", "macd_hist_norm", "stoch_k",
        # Volatility (3)
        "bb_width", "bb_position", "atr_change",
        # Trend (6)
        "adx", "di_plus", "di_minus", "close_ema21_dist", "ema_21_slope", "ema_50_200_spread",
        # Volume (2)
        "obv_momentum", "volume_roc",
        # Candle (1)
        "candle_body"
    ]
    
    # CRITICAL FIX: Apply final safety clipping to all features
    # This prevents any extreme outliers from breaking the model
    print("\nApplying safety clipping to features...")
    for col in feature_cols:
        if col in df_indexed.columns:
            # Get reasonable bounds based on training data ranges
            q01 = df_indexed[col].quantile(0.01)
            q99 = df_indexed[col].quantile(0.99)
            iqr = q99 - q01
            lower_bound = q01 - 10 * iqr  # Allow 10x IQR below 1st percentile
            upper_bound = q99 + 10 * iqr  # Allow 10x IQR above 99th percentile
            
            # Clip to reasonable bounds
            df_indexed[col] = df_indexed[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"✓ Added {len(feature_cols)} technical indicators")
    print(f"Features: {', '.join(feature_cols[:5])}... (showing first 5)")
    
    # Drop rows with NaN
    initial_len = len(df_indexed)
    df_indexed = df_indexed.dropna()
    dropped = initial_len - len(df_indexed)
    
    if dropped > 0:
        print(f"✓ Dropped {dropped} rows with NaN values")
    
    # Reset index to get Time back as column
    df_with_indicators = df_indexed.reset_index()
    
    print(f"✓ Final dataset: {len(df_with_indicators):,} bars")
    
    return df_with_indicators, feature_cols


def save_data(df, filename="live_eurusd_data.csv"):
    """Save data to CSV with Time as index (matching training format)"""
    filepath = os.path.join("live_testing", filename)
    
    # Set Time as index to match training data format
    if 'Time' in df.columns:
        df = df.set_index('Time')
    
    df.to_csv(filepath)
    print(f"\n✓ Data saved to: {filepath}")
    print(f"✓ Time set as index (matching training format)")
    return filepath


def main():
    print("\n" + "=" * 60)
    print("MT5 LIVE DATA FETCHER")
    print("=" * 60)
    print("This script will:")
    print("1. Connect to MetaTrader 5")
    print("2. Fetch maximum available EUR/USD H1 data")
    print("3. Add technical indicators")
    print("4. Save prepared data for testing")
    print()
    
    # Fetch data
    df = fetch_eurusd_data(timeframe=mt5.TIMEFRAME_H1, bars=10000)
    
    if df is None:
        print("\n✗ Failed to fetch data from MT5")
        print("\nTroubleshooting:")
        print("1. Make sure MetaTrader 5 is installed and running")
        print("2. Check that you're logged into an account")
        print("3. Verify EUR/USD is available in Market Watch")
        return
    
    # Prepare data
    df_prepared, feature_cols = prepare_data_for_testing(df)
    
    # Save data
    filepath = save_data(df_prepared)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total bars fetched: {len(df_prepared):,}")
    print(f"Date range: {df_prepared['Time'].min()} to {df_prepared['Time'].max()}")
    print(f"Features: {len(feature_cols)}")
    print(f"File: {filepath}")
    print("\n✓ Ready for testing!")
    print("\nNext step: Run 'python live_testing/test_on_live_data.py'")


if __name__ == "__main__":
    main()
