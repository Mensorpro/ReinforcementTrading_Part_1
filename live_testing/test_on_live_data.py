"""
Test the trained model on live MT5 data
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import ForexTradingEnv


def load_live_data():
    """Load the prepared live data"""
    filepath = "live_testing/live_eurusd_data.csv"
    
    if not os.path.exists(filepath):
        print(f"✗ Data file not found: {filepath}")
        print("\nPlease run 'python live_testing/fetch_mt5_data.py' first")
        return None, None
    
    # Load with Time as index (matching training format)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Use ONLY the same 29 features as training (in the same order)
    feature_cols = [
        "rsi_14", "rsi_9", "macd_norm", "macd_hist_norm", "stoch_k", "stoch_d", "roc_10",
        "bb_width", "bb_position", "atr_change",
        "adx", "di_plus", "di_minus",
        "close_ema9_dist", "close_ema21_dist", "close_ema50_dist", "close_ema200_dist",
        "ema_9_slope", "ema_21_slope", "ema_50_slope",
        "ema_9_21_spread", "ema_21_50_spread", "ema_50_200_spread",
        "obv_momentum", "volume_roc",
        "candle_body", "candle_range", "upper_wick", "lower_wick"
    ]
    
    # Check if all required features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        print(f"✗ Missing features in data: {missing_features}")
        return None, None
    
    print("=" * 60)
    print("LIVE DATA LOADED")
    print("=" * 60)
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Features: {len(feature_cols)} (matching training)")
    print(f"Latest close: {df['Close'].iloc[-1]:.5f}")
    print(f"Index type: {type(df.index).__name__} (matching training)")
    print()
    
    return df, feature_cols


def load_trained_model():
    """Load the best trained model"""
    model_path = "model_eurusd_best.zip"
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        print("\nPlease train the model first using 'python train_agent.py'")
        return None
    
    print("=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)
    print(f"Model: {model_path}")
    
    # Create dummy env with all 29 features (same as training)
    feature_cols = [
        "rsi_14", "rsi_9", "macd_norm", "macd_hist_norm", "stoch_k", "stoch_d", "roc_10",
        "bb_width", "bb_position", "atr_change",
        "adx", "di_plus", "di_minus",
        "close_ema9_dist", "close_ema21_dist", "close_ema50_dist", "close_ema200_dist",
        "ema_9_slope", "ema_21_slope", "ema_50_slope",
        "ema_9_21_spread", "ema_21_50_spread", "ema_50_200_spread",
        "obv_momentum", "volume_roc",
        "candle_body", "candle_range", "upper_wick", "lower_wick"
    ]
    
    dummy_data = {
        'Open': [1.0] * 50, 
        'High': [1.0] * 50, 
        'Low': [1.0] * 50, 
        'Close': [1.0] * 50,
        'atr_14': [0.001] * 50
    }
    # Add all feature columns with dummy values
    for col in feature_cols:
        dummy_data[col] = [0.0] * 50
    
    dummy_df = pd.DataFrame(dummy_data)
    dummy_env = DummyVecEnv([lambda: ForexTradingEnv(
        df=dummy_df, 
        window_size=30, 
        feature_columns=feature_cols
    )])
    
    model = PPO.load(model_path, env=dummy_env)
    print("✓ Model loaded successfully")
    print()
    
    return model


def create_test_env(df, feature_cols):
    """Create testing environment with live data"""
    def make_env():
        return ForexTradingEnv(
            df=df,
            window_size=30,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            open_penalty_pips=2.0,
            time_penalty_pips=0.05,
            atr_sl_multiplier=1.5,
            atr_tp_multiplier=3.0
        )
    
    return DummyVecEnv([make_env])


def test_model(model, test_env):
    """Run the model on test environment and collect results"""
    print("=" * 60)
    print("TESTING MODEL ON LIVE DATA")
    print("=" * 60)
    print("Running simulation...")
    
    obs = test_env.reset()
    equity_curve = [10000.0]  # Starting equity
    trades = []
    actions_taken = []
    
    step_count = 0
    
    while True:
        # Get model prediction
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action[0]))
        
        # Take step
        step_out = test_env.step(action)
        
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        
        # Get info
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity = info.get("equity_usd", test_env.get_attr("equity_usd")[0])
        equity_curve.append(equity)
        
        # Track trades
        trade_info = info.get("last_trade_info")
        if trade_info and trade_info.get("event") == "CLOSE":
            trades.append(trade_info)
        
        step_count += 1
        
        if done:
            break
    
    print(f"✓ Simulation complete: {step_count:,} steps")
    print()
    
    return equity_curve, trades, actions_taken


def calculate_metrics(equity_curve, trades):
    """Calculate performance metrics"""
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Total return
    total_return = ((equity_array[-1] - equity_array[0]) / equity_array[0]) * 100
    
    # Sharpe Ratio (annualized)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0
    
    # Max Drawdown
    cummax = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - cummax) / cummax
    max_drawdown = np.min(drawdown) * 100
    
    # Trade statistics
    if trades:
        winning_trades = [t for t in trades if t.get("net_pips", 0) > 0]
        losing_trades = [t for t in trades if t.get("net_pips", 0) <= 0]
        
        win_rate = (len(winning_trades) / len(trades)) * 100
        
        avg_win = np.mean([t["net_pips"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["net_pips"] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = (sum([t["net_pips"] for t in winning_trades]) / 
                        abs(sum([t["net_pips"] for t in losing_trades]))) if losing_trades else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return {
        "final_equity": float(equity_array[-1]),
        "total_return_pct": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_drawdown),
        "num_trades": len(trades),
        "win_rate": float(win_rate),
        "avg_win_pips": float(avg_win),
        "avg_loss_pips": float(avg_loss),
        "profit_factor": float(profit_factor)
    }


def print_results(metrics, trades):
    """Print test results"""
    print("=" * 60)
    print("LIVE DATA TEST RESULTS")
    print("=" * 60)
    print()
    print("PERFORMANCE METRICS:")
    print(f"  Final Equity       : ${metrics['final_equity']:,.2f}")
    print(f"  Total Return       : {metrics['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio       : {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown_pct']:.2f}%")
    print()
    print("TRADING STATISTICS:")
    print(f"  Total Trades       : {metrics['num_trades']}")
    print(f"  Win Rate           : {metrics['win_rate']:.1f}%")
    print(f"  Average Win        : {metrics['avg_win_pips']:+.2f} pips")
    print(f"  Average Loss       : {metrics['avg_loss_pips']:+.2f} pips")
    print(f"  Profit Factor      : {metrics['profit_factor']:.2f}")
    print()
    
    # Show last 5 trades
    if trades:
        print("LAST 5 TRADES:")
        for trade in trades[-5:]:
            reason = trade.get("reason", "UNKNOWN")
            net_pips = trade.get("net_pips", 0)
            direction = "LONG" if trade.get("position", 0) == 1 else "SHORT"
            time_in_trade = trade.get("time_in_trade", 0)
            
            result = "WIN" if net_pips > 0 else "LOSS"
            print(f"  {direction:5} | {reason:20} | {net_pips:+7.2f} pips | {time_in_trade:3} bars | {result}")
    print()


def plot_results(equity_curve, df, trades):
    """Create visualization of results"""
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Equity Curve
    ax1 = axes[0]
    ax1.plot(equity_curve, linewidth=2, color='blue')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Equity')
    ax1.fill_between(range(len(equity_curve)), 10000, equity_curve, 
                     where=np.array(equity_curve) >= 10000, alpha=0.3, color='green', label='Profit')
    ax1.fill_between(range(len(equity_curve)), 10000, equity_curve,
                     where=np.array(equity_curve) < 10000, alpha=0.3, color='red', label='Loss')
    ax1.set_title("Equity Curve on Live Data", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Price with Trade Markers
    ax2 = axes[1]
    # Use reset_index to get integer positions for plotting
    price_data = df['Close'].reset_index(drop=True).values[:len(equity_curve)]
    ax2.plot(price_data, linewidth=1, color='black', alpha=0.7, label='EUR/USD Price')
    
    # Mark trades
    for trade in trades:
        step = trade.get("step", 0)
        if step < len(price_data):
            color = 'green' if trade.get("net_pips", 0) > 0 else 'red'
            marker = '^' if trade.get("position", 0) == 1 else 'v'
            ax2.scatter(step, price_data[step], color=color, marker=marker, s=50, alpha=0.6)
    
    ax2.set_title("EUR/USD Price with Trade Markers (Green=Win, Red=Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    equity_array = np.array(equity_curve)
    cummax = np.maximum.accumulate(equity_array)
    drawdown = ((equity_array - cummax) / cummax) * 100
    ax3.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax3.plot(drawdown, color='darkred', linewidth=1)
    ax3.set_title("Drawdown (%)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filepath = "live_testing/live_test_results.png"
    plt.savefig(filepath, dpi=150)
    print(f"✓ Plot saved: {filepath}")
    
    plt.show()


def main():
    print("\n" + "=" * 60)
    print("LIVE DATA MODEL TESTING")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    df, feature_cols = load_live_data()
    if df is None:
        return
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Create test environment
    test_env = create_test_env(df, feature_cols)
    
    # Run test
    equity_curve, trades, actions = test_model(model, test_env)
    
    # Calculate metrics
    metrics = calculate_metrics(equity_curve, trades)
    
    # Print results
    print_results(metrics, trades)
    
    # Plot results
    plot_results(equity_curve, df, trades)
    
    # Save detailed results
    results_file = "live_testing/test_results.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LIVE DATA TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Total Bars: {len(df):,}\n\n")
        f.write("PERFORMANCE METRICS:\n")
        for key, value in metrics.items():
            f.write(f"  {key:20}: {value}\n")
    
    print(f"✓ Detailed results saved: {results_file}")
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
