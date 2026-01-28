# trading_env.py

from __future__ import annotations

import numpy as np

# Prefer gymnasium if available (SB3 supports it), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class ForexTradingEnv(gym.Env):
    """
    RL Forex Trading Environment (Position-Persistent)

    Key properties:
      - Observation: rolling window of features + 3 state features (position, time_in_trade, unrealized_pnl_pips)
      - Actions:
          0: HOLD (do nothing)
          1: CLOSE (close position if any)
          2..: OPEN (direction + SL + TP), only effective when flat
      - Position persistence: once open, position remains until:
          - agent sends CLOSE, or
          - SL/TP hit intrabar
      - Friction: spread + commission + optional slippage
      - Reward:
          - realized PnL (pips) minus costs (pips) on closes
          - optional shaping via delta unrealized PnL (pips) while holding
      - Random episode start to reduce memorization / overfit
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 100,
        feature_columns = None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,              # cost in pips per round-trip (approx)
        commission_pips: float = 0.0,          # cost in pips per round-trip
        max_slippage_pips: float = 0.0,        # random extra pips (0..max) applied on fills
        lot_size: float = 100000.0,            # 1.0 lot = 100k units (for equity in $)
        reward_scale: float = 1.0,             # optional scaling of rewards
        unrealized_delta_weight: float = 0.02, # shaping weight on delta-unrealized while holding
        random_start: bool = True,
        min_episode_steps: int = 300,          # minimum steps per episode (for random starts)
        episode_max_steps: int | None = None,  # optional cap (truncation)
        feature_mean: np.ndarray | None = None, # optional normalization (train-fitted)
        feature_std: np.ndarray | None = None,  # optional normalization (train-fitted)
        allow_flip: bool = False,               # if True, OPEN while in position flips (close+open). Default False.
        hold_reward_weight: float = 0.005,   # tuned below
        open_penalty_pips: float = 0.5,      # NEW: penalty per open
        time_penalty_pips: float = 0.02,     # NEW: cost per bar in a trade
        atr_sl_multiplier: float = 1.5,      # SL = ATR × this multiplier
        atr_tp_multiplier: float = 3.0,      # TP = ATR × this multiplier (2:1 risk-reward)
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)  # fallback: everything
        else:
            self.feature_columns = list(feature_columns)

        # Ensure ATR is in the dataframe for dynamic SL/TP
        if 'atr_14' not in self.df.columns:
            raise ValueError("DataFrame must contain 'atr_14' column for ATR-based SL/TP")

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe is too short for the given window_size.")

        self.window_size = int(window_size)
        self.pip_value = float(pip_value)

        # ATR-based SL/TP multipliers
        self.atr_sl_multiplier = float(atr_sl_multiplier)
        self.atr_tp_multiplier = float(atr_tp_multiplier)

        # Friction
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        # Equity accounting (approx): for EURUSD 1 pip per 1 lot ≈ $10
        # pip_value (price) * lot_size (units) ≈ $ per 1.0 price move.
        # 1 pip = pip_value price move, so $/pip ≈ pip_value * lot_size.
        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        # Reward handling
        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)

        # Episode handling
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        # Optional normalization (fit on train only, pass arrays here)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.allow_flip = bool(allow_flip)

        # --- Actions (Simplified to 3 actions - NO MANUAL CLOSE) ---
        # 0: HOLD
        # 1: LONG (SL/TP calculated dynamically from ATR)
        # 2: SHORT (SL/TP calculated dynamically from ATR)
        # Removed CLOSE action to force proper risk-reward ratio
        self.action_map = [
            ("HOLD", None),
            ("LONG", 1),   # direction: 1 = long
            ("SHORT", -1)  # direction: -1 = short
        ]

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation features: df columns + 4 state features (Account State)
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 4  # position, time_in_trade, unrealized_pnl, drawdown
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        # Internal state
        self._reset_state()

    # ----------------------------
    # Core Helpers
    # ----------------------------

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position state
        self.position = 0              # 0=flat, +1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.max_unrealized_pips = 0.0  # Track peak unrealized profit for drawdown penalty
        self.sl_pips = 0.0  # Store SL distance for risk-adjusted returns

        # Accounting
        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.peak_equity_usd = self.initial_equity_usd  # Track peak for drawdown calculation

        # Logging
        self.equity_curve = []
        self.last_trade_info = None

    def _get_state_features(self):
        """
        Account State features (The "Health" - Professional PPO Strategy):
        1. Position: -1 (short), 0 (flat), +1 (long)
        2. Time in trade: normalized by typical trade duration (100 bars)
        3. Unrealized PnL: normalized by risk (SL distance in pips) - "R multiple"
        4. Drawdown: current distance from peak equity [0, 1]
        
        These features tell the agent about its "health" and risk exposure.
        """
        # 1. Position: already in [-1, 0, 1]
        pos = float(self.position)
        
        # 2. Time in trade: normalize by typical duration (100 bars), clip to [0, 1]
        t_norm = np.clip(float(self.time_in_trade) / 100.0, 0.0, 1.0)
        
        # 3. Unrealized PnL: normalize by risk taken (SL distance)
        if self.position != 0 and self.sl_pips > 0:
            unreal_pips = float(self._compute_unrealized_pips())
            # Express as multiple of risk: +1.0 = +1R (at TP), -1.0 = -1R (at SL)
            unreal_scaled = np.clip(unreal_pips / self.sl_pips, -2.0, 2.0)
        else:
            unreal_scaled = 0.0
        
        # 4. Drawdown: (PeakEquity - CurrentEquity) / InitialBalance ∈ [0, 1]
        # This tells the agent how much it's "down" from its peak
        drawdown = (self.peak_equity_usd - self.equity_usd) / self.initial_equity_usd
        drawdown_norm = np.clip(drawdown, 0.0, 1.0)  # 0 = at peak, 1 = lost 100%
        
        return np.array([pos, t_norm, unreal_scaled, drawdown_norm], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        if self.position == 1:
            pnl_price = close_price - self.entry_price
        else:
            pnl_price = self.entry_price - close_price
        return pnl_price / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, 1, -1)
        std = self.feature_std.reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        start = self.current_step - self.window_size
        if start < 0:
            start = 0

        obs_df = self.df.iloc[start:self.current_step].copy()
        # use only selected feature columns for the agent
        obs_df = obs_df[self.feature_columns]

        # If empty (safety), use the first row repeated
        if len(obs_df) == 0:
            base = np.tile(self.df.iloc[0].values.astype(np.float32), (self.window_size, 1))
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                pad = np.tile(base[0], (pad_rows, 1))
                base = np.vstack([pad, base])

        # Append state features (same for each row)
        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)

        # Optional normalization (only if user passes train-fitted mean/std matching obs dims)
        obs = self._apply_optional_normalization(obs)

        return obs

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        # Simple friction model (round-trip)
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int):
        # Get current ATR for dynamic SL/TP calculation
        current_atr = float(self.df.loc[self.current_step, "atr_14"])
        
        # Calculate SL and TP in pips based on ATR
        sl_pips = (current_atr / self.pip_value) * self.atr_sl_multiplier
        tp_pips = (current_atr / self.pip_value) * self.atr_tp_multiplier
        
        # Entry on current close + slippage; costs applied on close (round-trip model)
        close_price = float(self.df.loc[self.current_step, "Close"])
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:  # long
            entry = close_price + slip_price
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:               # short
            entry = close_price - slip_price
            sl_price = entry + sl_pips * self.pip_value
            tp_price = entry - tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.max_unrealized_pips = 0.0  # Reset for new trade
        self.sl_pips = sl_pips  # Store for risk-adjusted calculations

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "atr": current_atr,
            "sl_pips": float(sl_pips),
            "tp_pips": float(tp_pips)
        }

    def _close_position(self, reason: str, exit_price: float):
        # Realized pips
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        # Costs in pips (round-trip)
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips

        # Update equity in USD
        self.equity_usd += net_pips * self.usd_per_pip
        
        # Update peak equity for drawdown calculation
        if self.equity_usd > self.peak_equity_usd:
            self.peak_equity_usd = self.equity_usd

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
        }

        # Reset position state
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.max_unrealized_pips = 0.0
        self.sl_pips = 0.0

        self.last_trade_info = trade_info
        return net_pips

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float:
        """
        Checks SL/TP on the *current bar* range [Low, High].
        Conservative rule if both touched: assume SL hits first (worst case).
        Returns realized net pips if closed; otherwise None.
        """
        if self.position == 0:
            return None

        # If last bar, close on close
        if self.current_step >= self.n_steps - 1:
            exit_price = float(self.df.loc[self.current_step, "Close"])
            net_pips = self._close_position("END_OF_DATA", exit_price)
            return net_pips

        # Check CURRENT bar's High/Low (not next bar!)
        current_high = float(self.df.loc[self.current_step, "High"])
        current_low = float(self.df.loc[self.current_step, "Low"])

        if self.position == 1:
            sl_hit = current_low <= self.sl_price
            tp_hit = current_high >= self.tp_price
            if sl_hit and tp_hit:
                # conservative: SL first
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            sl_hit = current_high >= self.sl_price
            tp_hit = current_low <= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ----------------------------
    # Gym API
    # ----------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_state()

        # Choose start
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(np.random.randint(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()

        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            # If someone steps after done, just return current obs with 0 reward
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1

        # Clear last trade info at start of step (prevents ghost trades)
        self.last_trade_info = None

        # Reward components (PPO-optimized system with tuned coefficients)
        reward = 0.0
        info = {}

        act_type, direction = self.action_map[int(action)]

        # 1) Apply action logic
        if act_type == "HOLD":
            pass

        elif act_type in ["LONG", "SHORT"]:
            if self.position == 0:
                # SIMPLIFIED REWARD: Only trend-following bonus/penalty at entry
                ema_21_slope = float(self.df.loc[self.current_step, "ema_21_slope_norm"])
                ema_50_200_spread = float(self.df.loc[self.current_step, "ema_50_200_spread_norm"])
                
                # Reward trading WITH the trend, penalize trading AGAINST it
                if direction == 1:  # LONG entry
                    if ema_21_slope > 0.1 and ema_50_200_spread > 0.1:  # Strong uptrend
                        reward += 0.05  # Bonus for trend-following
                    elif ema_21_slope < -0.1 or ema_50_200_spread < -0.1:  # Counter-trend
                        reward -= 0.08  # Penalty for counter-trend
                elif direction == -1:  # SHORT entry
                    if ema_21_slope < -0.1 and ema_50_200_spread < -0.1:  # Strong downtrend
                        reward += 0.05  # Bonus for trend-following
                    elif ema_21_slope > 0.1 or ema_50_200_spread > 0.1:  # Counter-trend
                        reward -= 0.08  # Penalty for counter-trend
                
                self._open_position(direction=direction)
            else:
                if self.allow_flip:
                    close_price = float(self.df.loc[self.current_step, "Close"])
                    net_pips = self._close_position("FLIP_CLOSE", close_price)
                    
                    # Simple PnL reward (no log scaling, no trading cost penalty)
                    reward += net_pips / 100.0
                    
                    # Trend-following check for flip
                    ema_21_slope = float(self.df.loc[self.current_step, "ema_21_slope_norm"])
                    ema_50_200_spread = float(self.df.loc[self.current_step, "ema_50_200_spread_norm"])
                    
                    if direction == 1:
                        if ema_21_slope > 0.1 and ema_50_200_spread > 0.1:
                            reward += 0.05
                        elif ema_21_slope < -0.1 or ema_50_200_spread < -0.1:
                            reward -= 0.08
                    elif direction == -1:
                        if ema_21_slope < -0.1 and ema_50_200_spread < -0.1:
                            reward += 0.05
                        elif ema_21_slope > 0.1 or ema_50_200_spread > 0.1:
                            reward -= 0.08
                    
                    self._open_position(direction=direction)

        # 2) Advance time FIRST
        self.current_step += 1

        # 3) THEN check SL/TP on the new current bar (no future data!)
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            # SIMPLIFIED: Just reward net PnL (no log scaling, no penalties)
            reward += realized_now / 100.0

        # 4) If still open, update unrealized tracking (for observations only, NO reward penalty)
        if self.position != 0:
            self.time_in_trade += 1
            
            # Update unrealized tracking (for info only, not reward)
            unreal_now = self._compute_unrealized_pips()
            if unreal_now > self.max_unrealized_pips:
                self.max_unrealized_pips = unreal_now
            self.prev_unrealized_pips = unreal_now

        # 5) Termination / truncation
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # 6) Log equity
        self.equity_curve.append(float(self.equity_usd))

        # 7) Build observation
        obs = self._get_observation()

        # 8) MANDATORY: Clip reward to [-1, +1] for PPO stability
        reward = float(np.clip(reward, -1.0, 1.0))

        # 9) Info
        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward": float(reward),
            "last_trade_info": self.last_trade_info
        })

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, reward, done, info

    def render(self):
        print(
            f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )
