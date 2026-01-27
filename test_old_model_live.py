"""
Quick test of OLD model on live data with OLD settings (4 actions)
This shows baseline performance before retraining
"""
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import ForexTradingEnv
from indicators import load_and_preprocess_data

print("Loading live data...")
df, cols = load_and_preprocess_data('live_testing/live_eurusd_data.csv')
print(f"✓ Loaded {len(df)} bars")

print("\nCreating environment with OLD settings (4 actions)...")
# Temporarily restore old action map for testing old model
env = ForexTradingEnv(
    df=df,
    window_size=30,
    spread_pips=1.0,
    commission_pips=0.0,
    max_slippage_pips=0.2,
    random_start=False,
    episode_max_steps=None,
    feature_columns=cols,
    open_penalty_pips=2.0,      # OLD settings
    time_penalty_pips=0.05,     # OLD settings
    atr_sl_multiplier=1.5,      # OLD settings
    atr_tp_multiplier=3.0       # OLD settings
)

# Manually restore 4-action map for old model
env.action_map = [
    ("HOLD", None),
    ("CLOSE", None),
    ("LONG", 1),
    ("SHORT", -1)
]
env.action_space.n = 4

vec_env = DummyVecEnv([lambda: env])

print("Loading OLD trained model...")
model = PPO.load("model_eurusd_best.zip", env=vec_env)
print("✓ Model loaded")

print("\nRunning simulation...")
obs = vec_env.reset()
equity_curve = [10000.0]
trades = []

step = 0
while step < len(df) - 30:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    
    equity = info[0].get("equity_usd", vec_env.get_attr("equity_usd")[0])
    equity_curve.append(equity)
    
    trade_info = info[0].get("last_trade_info")
    if trade_info and trade_info.get("event") == "CLOSE":
        trades.append(trade_info)
    
    step += 1
    if done[0]:
        break

print(f"✓ Simulation complete: {step} steps")

# Calculate metrics
final_equity = equity_curve[-1]
total_return = ((final_equity - 10000) / 10000) * 100

if trades:
    winning_trades = [t for t in trades if t.get("net_pips", 0) > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_win = sum([t["net_pips"] for t in winning_trades]) / len(winning_trades) if winning_trades else 0
    avg_loss = sum([t["net_pips"] for t in trades if t["net_pips"] <= 0]) / len([t for t in trades if t["net_pips"] <= 0]) if any(t["net_pips"] <= 0 for t in trades) else 0
    profit_factor = sum([t["net_pips"] for t in winning_trades]) / abs(sum([t["net_pips"] for t in trades if t["net_pips"] <= 0])) if any(t["net_pips"] <= 0 for t in trades) else 0
else:
    win_rate = 0
    avg_win = 0
    avg_loss = 0
    profit_factor = 0

print("\n" + "="*60)
print("OLD MODEL ON LIVE DATA (BASELINE)")
print("="*60)
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Total Return: {total_return:+.2f}%")
print(f"Total Trades: {len(trades)}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Avg Win: {avg_win:+.2f} pips")
print(f"Avg Loss: {avg_loss:+.2f} pips")
print(f"Profit Factor: {profit_factor:.2f}")
print("="*60)
print("\nThis is your BASELINE before retraining with new settings.")
