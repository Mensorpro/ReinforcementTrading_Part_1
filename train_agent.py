import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


class EntropyDecayCallback(BaseCallback):
    """
    Callback to gradually reduce entropy coefficient during training.
    Uses EXPONENTIAL decay for faster reduction early, slower late.
    This prevents catastrophic forgetting by reducing exploration quickly.
    
    Args:
        initial_ent_coef: Starting entropy coefficient (high exploration)
        final_ent_coef: Ending entropy coefficient (low exploration)
        decay_steps: Number of timesteps over which to decay
        verbose: Verbosity level
    """
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, decay_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_steps = decay_steps
        
    def _on_step(self) -> bool:
        # Calculate current entropy coefficient based on progress
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        
        # EXPONENTIAL decay: drops fast early, slow later (prevents catastrophic forgetting)
        current_ent_coef = self.final_ent_coef + (
            (self.initial_ent_coef - self.final_ent_coef) * (1 - progress) ** 2
        )
        
        # Update model's entropy coefficient
        self.model.ent_coef = current_ent_coef
        
        # Log every 50k steps
        if self.num_timesteps % 50000 == 0:
            if self.verbose > 0:
                print(f"\n[Entropy Decay] Step {self.num_timesteps:,}: ent_coef = {current_ent_coef:.6f} (progress: {progress*100:.1f}%)")
        
        return True


class ClipRangeDecayCallback(BaseCallback):
    """
    Callback to gradually reduce clip range during training.
    Uses EXPONENTIAL decay to make policy updates more conservative faster.
    This prevents catastrophic forgetting.
    
    Args:
        initial_clip_range: Starting clip range (larger updates)
        final_clip_range: Ending clip range (smaller updates)
        decay_steps: Number of timesteps over which to decay
        verbose: Verbosity level
    """
    def __init__(self, initial_clip_range: float, final_clip_range: float, decay_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_clip_range = initial_clip_range
        self.final_clip_range = final_clip_range
        self.decay_steps = decay_steps
        
    def _on_step(self) -> bool:
        # Calculate current clip range based on progress
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        
        # EXPONENTIAL decay: drops fast early, slow later
        current_clip_range = self.final_clip_range + (
            (self.initial_clip_range - self.final_clip_range) * (1 - progress) ** 2
        )
        
        # Update model's clip range (need to create a lambda that returns the value)
        self.model.clip_range = lambda _: current_clip_range
        
        # Log every 50k steps
        if self.num_timesteps % 50000 == 0:
            if self.verbose > 0:
                print(f"[Clip Range Decay] Step {self.num_timesteps:,}: clip_range = {current_clip_range:.4f}")
        
        return True


def calculate_metrics(equity_curve):
    """Calculate trading performance metrics"""
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Sharpe Ratio (annualized, assuming hourly data)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized
    else:
        sharpe = 0.0
    
    # Max Drawdown
    cummax = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - cummax) / cummax
    max_drawdown = np.min(drawdown) * 100  # As percentage
    
    # Total Return
    total_return = ((equity_array[-1] - equity_array[0]) / equity_array[0]) * 100
    
    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_drawdown),
        "total_return_pct": float(total_return),
        "final_equity": float(equity_array[-1])
    }


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []
    trades = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Track action distribution

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_counts[int(action[0])] += 1  # Count this action
        
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # use equity from info (state *before* DummyVecEnv reset)
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)
        
        # Track closed trades
        trade_info = info.get("last_trade_info")
        if trade_info and trade_info.get("event") == "CLOSE":
            trades.append(trade_info)

        if done:
            break

    metrics = calculate_metrics(equity_curve)
    
    # Calculate win rate from trades
    if trades:
        winning_trades = sum(1 for t in trades if t.get("net_pips", 0) > 0)
        metrics["win_rate"] = (winning_trades / len(trades)) * 100
        metrics["num_trades"] = len(trades)
    else:
        metrics["win_rate"] = 0.0
        metrics["num_trades"] = 0
    
    # Add action distribution
    total_actions = sum(action_counts.values())
    metrics["action_distribution"] = {
        "HOLD": (action_counts[0] / total_actions * 100) if total_actions > 0 else 0,
        "CLOSE": (action_counts[1] / total_actions * 100) if total_actions > 0 else 0,
        "LONG": (action_counts[2] / total_actions * 100) if total_actions > 0 else 0,
        "SHORT": (action_counts[3] / total_actions * 100) if total_actions > 0 else 0,
    }
    
    return equity_curve, metrics



def main():
    file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Time split: 70% train, 15% validation, 15% test
    train_idx = int(len(df) * 0.7)
    val_idx = int(len(df) * 0.85)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()

    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    print(f"Training bars  : {len(train_df):,}")
    print(f"Validation bars: {len(val_df):,}")
    print(f"Testing bars   : {len(test_df):,}")
    print()

    # ---- Env factories ----
    WIN = 30

    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=200,
            episode_max_steps=None,
            feature_columns=feature_cols,
            open_penalty_pips=5.0,          # Increased from 2.0 to reduce overtrading
            time_penalty_pips=0.1,          # Increased from 0.05 to encourage faster exits
            atr_sl_multiplier=1.0,          # Tighter stop (was 1.5)
            atr_tp_multiplier=2.0           # Closer target (was 3.0) - 1:2 risk-reward
        )

    # Train-eval env: deterministic start, NO random starts (so curve is stable/reproducible)
    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            open_penalty_pips=5.0,
            time_penalty_pips=0.1,
            atr_sl_multiplier=1.0,
            atr_tp_multiplier=2.0
        )

    # Test-eval env: deterministic
    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            open_penalty_pips=5.0,
            time_penalty_pips=0.1,
            atr_sl_multiplier=1.0,
            atr_tp_multiplier=2.0
        )
    
    # Validation env: for early stopping
    def make_val_env():
        return ForexTradingEnv(
            df=val_df,
            window_size=WIN,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            open_penalty_pips=5.0,
            time_penalty_pips=0.1,
            atr_sl_multiplier=1.0,
            atr_tp_multiplier=2.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    val_eval_env = DummyVecEnv([make_val_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # ---- Model with Tuned Hyperparameters ----
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    
    # Entropy decay: Start lower, decay faster to prevent catastrophic forgetting
    initial_ent_coef = 0.02   # Lower exploration early (was 0.03)
    final_ent_coef = 0.0001   # Much lower exploration late (was 0.001)
    
    # Clip range decay: Start lower, end much lower for conservative updates
    initial_clip_range = 0.15 # Lower clip range (was 0.2)
    final_clip_range = 0.02   # Very conservative updates late (was 0.05)
    
    # Learning rate: Start lower to prevent overtraining
    initial_lr = 1e-4         # Lower learning rate (was 3e-4)
    final_lr = 1e-6           # Very low learning rate late (was 1e-5)
    
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=initial_lr,        # Will decay during training
        n_steps=4096,                    # Increased from 2048 for better value estimates
        batch_size=64,                   # Keep same
        n_epochs=10,                     # Keep same
        gamma=0.99,                      # Keep same
        gae_lambda=0.95,                 # Keep same
        clip_range=initial_clip_range,   # Will decay during training
        ent_coef=initial_ent_coef,       # Will decay during training
        vf_coef=2.0,                     # Increased to prioritize value function learning
        max_grad_norm=0.5,               # Keep same
        device='cpu',                    # MlpPolicy runs faster on CPU than GPU
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 64],            # Policy network: 2 layers (reduced to combat overfitting)
                vf=[128, 128, 64]        # Value network: 3 layers (reduced to combat overfitting)
            )
        ),
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )
    
    print(f"Learning rate    : {initial_lr} -> {final_lr} (decaying)")
    print(f"Policy network   : {model.policy_kwargs['net_arch']['pi']}")
    print(f"Value network    : {model.policy_kwargs['net_arch']['vf']}")
    print(f"N steps          : {model.n_steps}")
    print(f"Batch size       : {model.batch_size}")
    print(f"Entropy coef     : {initial_ent_coef} -> {final_ent_coef} (decaying)")
    print(f"Clip range       : {initial_clip_range} -> {final_clip_range} (decaying)")
    print(f"Value coef       : {model.vf_coef}")
    print()

    # ---- Callbacks ----
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Create tensorboard log directory
    os.makedirs("./tensorboard_log", exist_ok=True)

    # Checkpoint every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="ppo_eurusd"
    )
    
    total_timesteps = 500_000    # Reduced from 2M to prevent overtraining
    decay_steps = 200_000        # Complete decay in first 200k steps (faster)
    
    # Entropy decay: reduce exploration over time
    entropy_decay_callback = EntropyDecayCallback(
        initial_ent_coef=initial_ent_coef,
        final_ent_coef=final_ent_coef,
        decay_steps=decay_steps,
        verbose=1
    )
    
    # Clip range decay: make policy updates more conservative over time
    clip_range_decay_callback = ClipRangeDecayCallback(
        initial_clip_range=initial_clip_range,
        final_clip_range=final_clip_range,
        decay_steps=decay_steps,
        verbose=1
    )
    
    # Learning rate schedule: Warmup -> Plateau -> Decay
    # This prevents catastrophic forgetting by keeping LR stable when strategy is good
    class LearningRateScheduleCallback(BaseCallback):
        def __init__(self, initial_lr, final_lr, verbose=0):
            super().__init__(verbose)
            self.initial_lr = initial_lr
            self.final_lr = final_lr
            
        def _on_step(self):
            if self.num_timesteps < 50_000:
                # Warmup: 1e-5 -> 1e-4 (first 50k steps)
                progress = self.num_timesteps / 50_000
                current_lr = 1e-5 + (self.initial_lr - 1e-5) * progress
            elif self.num_timesteps < 150_000:
                # Plateau: keep at 1e-4 (50k-150k steps)
                # This prevents forgetting when model finds good strategy
                current_lr = self.initial_lr
            else:
                # Decay: 1e-4 -> 1e-6 (150k-500k steps)
                progress = min((self.num_timesteps - 150_000) / 350_000, 1.0)
                # Exponential decay for smoother transition
                current_lr = self.final_lr + (self.initial_lr - self.final_lr) * (1 - progress) ** 2
            
            self.model.learning_rate = current_lr
            
            if self.num_timesteps % 50000 == 0:
                if self.verbose > 0:
                    phase = "warmup" if self.num_timesteps < 50_000 else "plateau" if self.num_timesteps < 150_000 else "decay"
                    print(f"[Learning Rate Schedule] Step {self.num_timesteps:,}: lr = {current_lr:.6f} ({phase})")
            return True
    
    lr_schedule_callback = LearningRateScheduleCallback(
        initial_lr=initial_lr,
        final_lr=final_lr,
        verbose=1
    )
    
    # Early stopping based on validation performance
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,  # Reduced from 30 to stop earlier
        min_evals=5,                   # Reduced from 15
        verbose=1
    )
    
    # Custom evaluation callback with action distribution
    class CustomEvalCallback(BaseCallback):
        def __init__(self, eval_env, eval_freq, best_model_save_path, 
                     log_path, stop_callback, verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.best_model_save_path = best_model_save_path
            self.log_path = log_path
            self.stop_callback = stop_callback
            self.best_mean_reward = -np.inf
            self.evaluations_since_best = 0
            self.eval_log_file = os.path.join(log_path, "evaluations.txt")
            
            os.makedirs(best_model_save_path, exist_ok=True)
            os.makedirs(log_path, exist_ok=True)
            
            # Create log file with header
            with open(self.eval_log_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("TRAINING EVALUATIONS LOG\n")
                f.write("="*80 + "\n\n")
            
        def _on_step(self):
            if self.n_calls % self.eval_freq == 0:
                # Run evaluation with action tracking
                _, metrics = evaluate_model(self.model, self.eval_env, deterministic=True)
                mean_reward = metrics['total_return_pct']
                
                # Format output
                output = f"\n{'='*60}\n"
                output += f"EVAL at {self.num_timesteps:,} steps:\n"
                output += f"  Return: {mean_reward:+.2f}%\n"
                output += f"  Trades: {metrics['num_trades']}\n"
                output += f"  Win Rate: {metrics['win_rate']:.1f}%\n"
                
                action_dist = metrics.get('action_distribution', {})
                if action_dist:
                    output += f"  Actions:\n"
                    output += f"    HOLD : {action_dist['HOLD']:5.1f}%\n"
                    output += f"    CLOSE: {action_dist['CLOSE']:5.1f}%\n"
                    output += f"    LONG : {action_dist['LONG']:5.1f}%\n"
                    output += f"    SHORT: {action_dist['SHORT']:5.1f}%\n"
                
                # Check if best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.evaluations_since_best = 0
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    output += f"  ✓ New best model saved!\n"
                else:
                    self.evaluations_since_best += 1
                    output += f"  ({self.evaluations_since_best} evals since best)\n"
                
                output += f"{'='*60}\n"
                
                # Print to console
                print(output)
                
                # Append to log file
                with open(self.eval_log_file, 'a') as f:
                    f.write(output)
                
                # Check early stopping
                if self.stop_callback is not None:
                    if self.evaluations_since_best >= self.stop_callback.max_no_improvement_evals:
                        if self.n_calls // self.eval_freq >= self.stop_callback.min_evals:
                            stop_msg = f"\n⚠ Early stopping triggered after {self.evaluations_since_best} evals without improvement\n"
                            print(stop_msg)
                            with open(self.eval_log_file, 'a') as f:
                                f.write(stop_msg)
                            return False
            
            return True
    
    eval_callback = CustomEvalCallback(
        eval_env=val_eval_env,
        eval_freq=25_000,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        stop_callback=stop_callback,
        verbose=1
    )

    # ---- Train ----
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Total timesteps  : {total_timesteps:,}")
    print(f"Checkpoint freq  : 50,000 steps")
    print(f"Eval freq        : 25,000 steps")
    print(f"Early stopping   : Enabled (patience=10)")
    print()
    print("ANTI-OVERFITTING IMPROVEMENTS:")
    print("  • Reduced training: 500k steps (was 2M)")
    print("  • Smaller network: [128,64] policy, [128,128,64] value")
    print("  • Fewer features: 15 (was 29)")
    print("  • More episode diversity: 60 starting positions, no max length")
    print("  • Learning rate schedule: warmup (0-50k) -> plateau (50k-150k) -> decay (150k-500k)")
    print("  • Exponential entropy decay: 0.02 -> 0.0001 (fast reduction)")
    print("  • Exponential clip decay: 0.15 -> 0.02 (conservative updates)")
    print("  • Early stopping: 10 evals (was 30)")
    print()
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, lr_schedule_callback, entropy_decay_callback, clip_range_decay_callback, eval_callback]
    )

    # ---- Select best model by validation performance ----
    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)
    
    # Load best model from validation (saved by EvalCallback)
    best_model_path = "./best_model/best_model.zip"
    if os.path.exists(best_model_path):
        print("Loading best model from validation...")
        best_model = PPO.load(best_model_path, env=train_vec_env)
    else:
        print("No best model found, using final model...")
        best_model = model
    
    # Evaluate on all three sets
    print("\nEvaluating best model on all datasets...")
    
    equity_curve_train, metrics_train = evaluate_model(best_model, train_eval_env)
    equity_curve_val, metrics_val = evaluate_model(best_model, val_eval_env)
    equity_curve_test, metrics_test = evaluate_model(best_model, test_eval_env)
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    def print_metrics(name, metrics):
        print(f"\n{name}:")
        print(f"  Final Equity    : ${metrics['final_equity']:,.2f}")
        print(f"  Total Return    : {metrics['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio    : {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown    : {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate        : {metrics['win_rate']:.1f}%")
        print(f"  Number of Trades: {metrics['num_trades']}")
        
        # Print action distribution
        action_dist = metrics.get('action_distribution', {})
        if action_dist:
            print(f"  Action Distribution:")
            print(f"    HOLD : {action_dist['HOLD']:5.1f}%")
            print(f"    CLOSE: {action_dist['CLOSE']:5.1f}%")
            print(f"    LONG : {action_dist['LONG']:5.1f}%")
            print(f"    SHORT: {action_dist['SHORT']:5.1f}%")
    
    print_metrics("TRAIN SET", metrics_train)
    print_metrics("VALIDATION SET", metrics_val)
    print_metrics("TEST SET (Out-of-Sample)", metrics_test)
    
    # Save best model
    best_model.save("model_eurusd_best")
    print(f"\n✓ Best model saved: model_eurusd_best.zip")

    # ---- Plot Results ----
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Equity Curves
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve_train, label=f"Train (Return: {metrics_train['total_return_pct']:+.1f}%)", alpha=0.8)
    plt.plot(equity_curve_val, label=f"Validation (Return: {metrics_val['total_return_pct']:+.1f}%)", alpha=0.8)
    plt.plot(equity_curve_test, label=f"Test (Return: {metrics_test['total_return_pct']:+.1f}%)", alpha=0.8)
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Equity')
    plt.title("Equity Curves: Train / Validation / Test", fontsize=14, fontweight='bold')
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Returns Distribution
    plt.subplot(2, 1, 2)
    returns_train = np.diff(equity_curve_train) / np.array(equity_curve_train[:-1]) * 100
    returns_val = np.diff(equity_curve_val) / np.array(equity_curve_val[:-1]) * 100
    returns_test = np.diff(equity_curve_test) / np.array(equity_curve_test[:-1]) * 100
    
    plt.hist(returns_train, bins=50, alpha=0.5, label='Train', density=True)
    plt.hist(returns_val, bins=50, alpha=0.5, label='Validation', density=True)
    plt.hist(returns_test, bins=50, alpha=0.5, label='Test', density=True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.title("Returns Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Return (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("✓ Plot saved: training_results.png")
    plt.show()


if __name__ == "__main__":
    main()
