"""
Fetch live EUR/USD data from MetaTrader 5 and prepare it for testing
Uses the same preprocessing as training data (indicators.py)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import load_and_preprocess_data


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
    Add technical indicators using the SAME preprocessing as training
    This ensures live data matches training data format exactly
    
    Args:
        df: Raw OHLC DataFrame with Time column
    
    Returns:
        DataFrame with indicators, list of feature columns
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA (SAME AS TRAINING)")
    print("=" * 60)
    
    # Save to temporary CSV to use load_and_preprocess_data
    temp_file = "live_testing/temp_mt5_data.csv"
    
    # Rename Time column to match training data format
    df_renamed = df.rename(columns={'Time': 'Gmt time'})
    df_renamed.to_csv(temp_file, index=False)
    
    # Use the SAME preprocessing function as training
    df_processed, feature_cols = load_and_preprocess_data(temp_file)
    
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"✓ Processed {len(df_processed):,} bars")
    print(f"✓ Features: {len(feature_cols)} (19 market context)")
    print(f"✓ Includes: Normalized indicators + Temporal encoding")
    print(f"✓ Date range: {df_processed.index.min()} to {df_processed.index.max()}")
    
    return df_processed, feature_cols


def save_data(df, filename="live_eurusd_data.csv"):
    """Save data to CSV with Time as index (matching training format)"""
    filepath = os.path.join("live_testing", filename)
    
    # Data already has Time as index from load_and_preprocess_data
    df.to_csv(filepath)
    print(f"\n✓ Data saved to: {filepath}")
    print(f"✓ Time as index (matching training format)")
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
    print(f"Total bars: {len(df_prepared):,}")
    print(f"Date range: {df_prepared.index.min()} to {df_prepared.index.max()}")
    print(f"Features: {len(feature_cols)} (19 market context)")
    print(f"File: {filepath}")
    print("\n✓ Ready for testing!")
    print("\nNext step: Run 'python live_testing/test_on_live_data.py'")


if __name__ == "__main__":
    main()
