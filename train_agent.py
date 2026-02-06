import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, StopTrainingOnNoModelImprovement,
)

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


# ================================================================
# Helpers
# ================================================================

def calculate_metrics(equity_curve):
    eq = np.array(equity_curve)
    rets = np.diff(eq) / eq[:-1]
    sharpe = (
        np.mean(rets) / np.std(rets) * np.sqrt(252 * 24)
        if len(rets) > 1 and np.std(rets) > 0 else 0.0
    )
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax) / cummax
    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(np.min(dd) * 100),
        "total_return_pct": float((eq[-1] - eq[0]) / eq[0] * 100),
        "final_equity": float(eq[-1]),
    }


def evaluate_model(model, eval_env, deterministic=True):
    obs = eval_env.reset()
    equity_curve, trades = [], []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_counts[int(action[0])] += 1
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity_curve.append(info.get("equity_usd", eval_env.get_attr("equity_usd")[0]))

        ti = info.get("last_trade_info")
        if ti and ti.get("event") == "CLOSE":
            trades.append(ti)

        if done:
            break

    metrics = calculate_metrics(equity_curve)
    if trades:
        metrics["win_rate"] = sum(1 for t in trades if t.get("net_pips", 0) > 0) / len(trades) * 100
        metrics["num_trades"] = len(trades)
    else:
        metrics["win_rate"] = 0.0
        metrics["num_trades"] = 0

    total = sum(action_counts.values()) or 1
    metrics["action_distribution"] = {
        "HOLD":  action_counts[0] / total * 100,
        "LONG":  action_counts[1] / total * 100,
        "SHORT": action_counts[2] / total * 100,
        "CLOSE": action_counts[3] / total * 100,
    }
    return equity_curve, metrics


# ================================================================
# Custom eval callback (replaces over-engineered decay callbacks)
# ================================================================

class SimpleEvalCallback(BaseCallback):
    """Evaluate on validation set, save best, early-stop on plateau."""

    def __init__(self, eval_env, eval_freq, save_path, log_path,
                 patience=20, min_evals=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.log_path = log_path
        self.patience = patience
        self.min_evals = min_evals

        self.best_return = -np.inf
        self.evals_since_best = 0
        self.n_evals = 0

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        _, m = evaluate_model(self.model, self.eval_env, deterministic=True)
        self.n_evals += 1
        ret = m["total_return_pct"]

        ad = m.get("action_distribution", {})
        msg = (
            f"\n{'='*55}\n"
            f"EVAL #{self.n_evals}  step {self.num_timesteps:,}\n"
            f"  Return {ret:+.2f}%  |  Trades {m['num_trades']}  |  WR {m['win_rate']:.1f}%\n"
            f"  HOLD {ad.get('HOLD',0):.0f}%  LONG {ad.get('LONG',0):.0f}%  "
            f"SHORT {ad.get('SHORT',0):.0f}%  CLOSE {ad.get('CLOSE',0):.0f}%\n"
        )

        if ret > self.best_return:
            self.best_return = ret
            self.evals_since_best = 0
            self.model.save(os.path.join(self.save_path, "best_model"))
            msg += "  >> NEW BEST — saved\n"
        else:
            self.evals_since_best += 1
            msg += f"  ({self.evals_since_best}/{self.patience} patience)\n"

        msg += f"{'='*55}"
        print(msg)

        with open(os.path.join(self.log_path, "eval_log.txt"), "a") as f:
            f.write(msg + "\n")

        if self.n_evals >= self.min_evals and self.evals_since_best >= self.patience:
            print(f"\n*** Early stopping after {self.evals_since_best} evals without improvement ***")
            return False

        return True


# ================================================================
# Main
# ================================================================

def main():
    # ---- Data ----
    csv_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(csv_path)

    train_end = int(len(df) * 0.70)
    val_end   = int(len(df) * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print("=" * 60)
    print("DATA SPLIT")
    print(f"  Train : {len(train_df):,} bars")
    print(f"  Val   : {len(val_df):,} bars")
    print(f"  Test  : {len(test_df):,} bars")
    print(f"  Features per bar: {len(feature_cols)} market + 5 state = {len(feature_cols)+5}")
    print("=" * 60)

    # ---- Environment factories ----
    WARMUP = 30  # bars to skip at episode start (features are precomputed)

    def make_train():
        return ForexTradingEnv(
            df=train_df, feature_columns=feature_cols, window_size=WARMUP,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.2,
            random_start=True, min_episode_steps=300,
            randomize_costs=True,       # domain randomization for robustness
            atr_sl_multiplier=1.5, atr_tp_multiplier=3.0,
        )

    def make_train_eval():
        return ForexTradingEnv(
            df=train_df, feature_columns=feature_cols, window_size=WARMUP,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.2,
            random_start=False, randomize_costs=False,
            atr_sl_multiplier=1.5, atr_tp_multiplier=3.0,
        )

    def make_val():
        return ForexTradingEnv(
            df=val_df, feature_columns=feature_cols, window_size=WARMUP,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.2,
            random_start=False, randomize_costs=False,
            atr_sl_multiplier=1.5, atr_tp_multiplier=3.0,
        )

    def make_test():
        return ForexTradingEnv(
            df=test_df, feature_columns=feature_cols, window_size=WARMUP,
            spread_pips=1.0, commission_pips=0.0, max_slippage_pips=0.2,
            random_start=False, randomize_costs=False,
            atr_sl_multiplier=1.5, atr_tp_multiplier=3.0,
        )

    train_env      = DummyVecEnv([make_train])
    train_eval_env = DummyVecEnv([make_train_eval])
    val_eval_env   = DummyVecEnv([make_val])
    test_eval_env  = DummyVecEnv([make_test])

    # ---- PPO  (standard, no over-engineered decay schedules) ----
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128],
                vf=[256, 128],
            ),
        ),
        verbose=1,
        tensorboard_log="./tensorboard_log/",
    )

    print("\nMODEL")
    print(f"  Obs shape : {model.observation_space.shape}")
    print(f"  Actions   : 4 (HOLD, LONG, SHORT, CLOSE)")
    print(f"  Network   : pi={[256,128]}  vf={[256,128]}")
    print(f"  LR={3e-4}  ent={0.01}  clip={0.2}  gamma={0.99}")
    print()

    # ---- Callbacks ----
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./tensorboard_log", exist_ok=True)

    TOTAL_STEPS = 1_000_000
    EVAL_FREQ   = 25_000

    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints",
                                  name_prefix="ppo_eurusd")

    eval_cb = SimpleEvalCallback(
        eval_env=val_eval_env, eval_freq=EVAL_FREQ,
        save_path="./best_model/", log_path="./logs/",
        patience=20, min_evals=10,
    )

    print("TRAINING")
    print(f"  Total steps : {TOTAL_STEPS:,}")
    print(f"  Eval freq   : {EVAL_FREQ:,}")
    print(f"  Patience    : 20 evals")
    print(f"  Domain rand : ON (spread ±30%, slippage ±50%)")
    print()

    model.learn(total_timesteps=TOTAL_STEPS, callback=[ckpt_cb, eval_cb])

    # ---- Load best model ----
    best_path = "./best_model/best_model.zip"
    if os.path.exists(best_path):
        print("\nLoading best validation model...")
        best = PPO.load(best_path, env=train_env)
    else:
        print("\nNo best model file — using final model")
        best = model

    # ---- Evaluate on all sets ----
    eq_train, m_train = evaluate_model(best, train_eval_env)
    eq_val,   m_val   = evaluate_model(best, val_eval_env)
    eq_test,  m_test  = evaluate_model(best, test_eval_env)

    def _print(name, m):
        ad = m.get("action_distribution", {})
        print(f"  {name:12s}  Ret {m['total_return_pct']:+7.2f}%  "
              f"Sharpe {m['sharpe_ratio']:+.2f}  DD {m['max_drawdown_pct']:.1f}%  "
              f"Trades {m['num_trades']:4d}  WR {m['win_rate']:.0f}%  "
              f"H/L/S/C {ad.get('HOLD',0):.0f}/{ad.get('LONG',0):.0f}/"
              f"{ad.get('SHORT',0):.0f}/{ad.get('CLOSE',0):.0f}%")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    _print("Train", m_train)
    _print("Validation", m_val)
    _print("Test (OOS)", m_test)

    gap = abs(m_train["total_return_pct"] - m_test["total_return_pct"])
    if gap < 5:
        print("\n  Generalization looks GOOD (train-test gap < 5%)")
    elif gap < 15:
        print(f"\n  Moderate overfit (train-test gap = {gap:.1f}%)")
    else:
        print(f"\n  WARNING: likely overfit (train-test gap = {gap:.1f}%)")

    best.save("model_eurusd_best")
    print(f"\nModel saved: model_eurusd_best.zip")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(eq_train, label=f"Train ({m_train['total_return_pct']:+.1f}%)", alpha=.8)
    axes[0].plot(eq_val,   label=f"Val ({m_val['total_return_pct']:+.1f}%)",   alpha=.8)
    axes[0].plot(eq_test,  label=f"Test ({m_test['total_return_pct']:+.1f}%)",  alpha=.8)
    axes[0].axhline(10000, color="gray", ls="--", alpha=.4)
    axes[0].set_title("Equity Curves", fontweight="bold")
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend()
    axes[0].grid(alpha=.3)

    for eq, lbl in [(eq_train, "Train"), (eq_val, "Val"), (eq_test, "Test")]:
        r = np.diff(eq) / np.array(eq[:-1]) * 100
        axes[1].hist(r, bins=50, alpha=.4, label=lbl, density=True)
    axes[1].axvline(0, color="red", ls="--", alpha=.4)
    axes[1].set_title("Returns Distribution", fontweight="bold")
    axes[1].set_xlabel("Return (%)")
    axes[1].legend()
    axes[1].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("Plot saved: training_results.png")
    plt.show()


if __name__ == "__main__":
    main()
