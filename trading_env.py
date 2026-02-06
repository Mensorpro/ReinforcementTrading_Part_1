# trading_env.py

from __future__ import annotations

import numpy as np

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
    RL Forex Trading Environment — redesigned to learn internal market patterns.

    Key design principles (anti-overfitting):
      1. FLAT 1D observation: precomputed temporal features per bar + account state.
         No 2D window — MLP now receives semantically meaningful features.
      2. PURE PnL reward (R-multiples): no indicator-dependent bonuses.
         The agent MUST discover patterns from trading outcomes alone.
      3. 4 actions: HOLD, LONG, SHORT, CLOSE — agent controls entries AND exits.
      4. Domain randomization: costs vary each episode for robustness.

    Observation (1D vector):
      [38 market features] + [5 account state features] = 43 total

    Actions:
      0: HOLD  — do nothing
      1: LONG  — open long (if flat), else ignored
      2: SHORT — open short (if flat), else ignored
      3: CLOSE — close open position (if in trade), else ignored

    Reward:
      - On trade close: R-multiple (net_pips / sl_pips), clipped to [-3, 3]
      - While holding: tiny delta-unrealized shaping (in R units)
      - While flat: 0 (patience is free — prevents overtrading)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        feature_columns=None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,
        commission_pips: float = 0.0,
        max_slippage_pips: float = 0.2,
        lot_size: float = 100_000.0,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        randomize_costs: bool = False,
        # ---- kept for API compat but no longer used for obs shape ----
        window_size: int = 30,
        # ---- unused legacy params (accepted silently) ----
        **_kwargs,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)
        else:
            self.feature_columns = list(feature_columns)

        if 'atr_14' not in self.df.columns:
            raise ValueError("DataFrame must contain 'atr_14' for ATR-based SL/TP")

        self.pip_value = float(pip_value)
        self.atr_sl_multiplier = float(atr_sl_multiplier)
        self.atr_tp_multiplier = float(atr_tp_multiplier)

        # Costs (base values — may be randomized per episode)
        self._base_spread = float(spread_pips)
        self._base_commission = float(commission_pips)
        self._base_slippage = float(max_slippage_pips)
        self.spread_pips = self._base_spread
        self.commission_pips = self._base_commission
        self.max_slippage_pips = self._base_slippage
        self.randomize_costs = bool(randomize_costs)

        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        # Episode control
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = None if episode_max_steps is None else int(episode_max_steps)
        self.warmup_bars = int(window_size)  # bars to skip at start

        # Precompute numpy arrays for speed (avoid pandas in hot loop)
        self._close = self.df["Close"].values.astype(np.float64)
        self._high  = self.df["High"].values.astype(np.float64)
        self._low   = self.df["Low"].values.astype(np.float64)
        self._atr   = self.df["atr_14"].values.astype(np.float64)
        self._features = self.df[self.feature_columns].values.astype(np.float32)

        # --- Action space: 4 discrete actions ---
        self.action_map = [
            ("HOLD",  None),
            ("LONG",  1),
            ("SHORT", -1),
            ("CLOSE", None),
        ]
        self.action_space = spaces.Discrete(4)

        # --- Observation space: flat 1D vector ---
        self.n_market_features = len(self.feature_columns)
        self.n_state_features  = 5
        self.n_obs = self.n_market_features + self.n_state_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_obs,),
            dtype=np.float32,
        )

        self._reset_state()

    # ----------------------------------------------------------------
    # Internal state
    # ----------------------------------------------------------------

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position
        self.position = 0        # 0=flat, +1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.sl_pips = 0.0       # SL distance for R-multiple calc
        self.bars_since_last_trade = 0

        # Equity
        self.initial_equity = 10_000.0
        self.equity_usd = self.initial_equity
        self.peak_equity = self.initial_equity

        # Logging
        self.equity_curve = []
        self.last_trade_info = None

    # ----------------------------------------------------------------
    # Observation
    # ----------------------------------------------------------------

    def _get_state_features(self):
        """
        Account state (5 features):
          0  position           : {-1, 0, +1}
          1  time_in_trade_norm : time_in_trade / 100, clipped [0, 1]
          2  unrealized_R       : unrealized pips / sl_pips, clipped [-2, 2]
          3  drawdown_norm      : (peak - equity) / initial, clipped [0, 1]
          4  bars_flat_norm     : bars since last trade / 100, clipped [0, 1]
        """
        pos = float(self.position)

        tit = np.clip(self.time_in_trade / 100.0, 0.0, 1.0)

        if self.position != 0 and self.sl_pips > 0:
            ur = np.clip(self._compute_unrealized_pips() / self.sl_pips, -2.0, 2.0)
        else:
            ur = 0.0

        dd = np.clip((self.peak_equity - self.equity_usd) / self.initial_equity, 0.0, 1.0)

        bft = np.clip(self.bars_since_last_trade / 100.0, 0.0, 1.0) if self.position == 0 else 0.0

        return np.array([pos, tit, ur, dd, bft], dtype=np.float32)

    def _get_observation(self):
        market = self._features[self.current_step]          # (n_market_features,)
        state  = self._get_state_features()                 # (5,)
        return np.concatenate([market, state])

    # ----------------------------------------------------------------
    # Position helpers
    # ----------------------------------------------------------------

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        price = self._close[self.current_step]
        if self.position == 1:
            return (price - self.entry_price) / self.pip_value
        else:
            return (self.entry_price - price) / self.pip_value

    def _cost_pips(self):
        return self.spread_pips + self.commission_pips

    def _sample_slippage(self):
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _open_position(self, direction: int):
        atr = self._atr[self.current_step]
        sl_pips = (atr / self.pip_value) * self.atr_sl_multiplier
        tp_pips = (atr / self.pip_value) * self.atr_tp_multiplier

        close_price = self._close[self.current_step]
        slip = self._sample_slippage() * self.pip_value

        if direction == 1:
            entry = close_price + slip
            sl = entry - sl_pips * self.pip_value
            tp = entry + tp_pips * self.pip_value
        else:
            entry = close_price - slip
            sl = entry + sl_pips * self.pip_value
            tp = entry - tp_pips * self.pip_value

        self.position = direction
        self.entry_price = entry
        self.sl_price = sl
        self.tp_price = tp
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.sl_pips = sl_pips
        self.bars_since_last_trade = 0

        self.last_trade_info = {
            "event": "OPEN", "step": self.current_step,
            "position": self.position, "entry_price": entry,
            "sl_price": sl, "tp_price": tp,
            "sl_pips": float(sl_pips), "tp_pips": float(tp_pips),
        }

    def _close_position(self, reason: str, exit_price: float):
        if self.position == 1:
            pnl = (exit_price - self.entry_price) / self.pip_value
        else:
            pnl = (self.entry_price - exit_price) / self.pip_value

        cost = self._cost_pips()
        net = pnl - cost
        saved_sl = self.sl_pips          # save before reset

        self.equity_usd += net * self.usd_per_pip
        if self.equity_usd > self.peak_equity:
            self.peak_equity = self.equity_usd

        info = {
            "event": "CLOSE", "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(pnl),
            "cost_pips": float(cost),
            "net_pips": float(net),
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
        self.sl_pips = 0.0

        self.last_trade_info = info
        return net, saved_sl

    def _check_sl_tp(self) -> tuple | None:
        """Check SL/TP on current bar. Returns (net_pips, sl_pips) or None."""
        if self.position == 0:
            return None
        if self.current_step >= self.n_steps - 1:
            return self._close_position("END_OF_DATA", self._close[self.current_step])

        high = self._high[self.current_step]
        low  = self._low[self.current_step]

        if self.position == 1:
            sl_hit = low  <= self.sl_price
            tp_hit = high >= self.tp_price
        else:
            sl_hit = high >= self.sl_price
            tp_hit = low  <= self.tp_price

        if sl_hit and tp_hit:
            return self._close_position("SL_AND_TP_SAME_BAR", self.sl_price)
        if sl_hit:
            return self._close_position("SL_HIT", self.sl_price)
        if tp_hit:
            return self._close_position("TP_HIT", self.tp_price)
        return None

    # ----------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Domain randomization: vary costs each episode
        if self.randomize_costs:
            self.spread_pips     = np.random.uniform(self._base_spread * 0.7, self._base_spread * 1.5)
            self.commission_pips = self._base_commission
            self.max_slippage_pips = np.random.uniform(0.0, self._base_slippage * 1.5)
        else:
            self.spread_pips     = self._base_spread
            self.commission_pips = self._base_commission
            self.max_slippage_pips = self._base_slippage

        # Starting position
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.warmup_bars) - 2
            if max_start <= self.warmup_bars:
                self.current_step = self.warmup_bars
            else:
                self.current_step = int(np.random.randint(self.warmup_bars, max_start))
        else:
            self.current_step = self.warmup_bars

        obs = self._get_observation()
        return (obs, {}) if _GYMNASIUM else obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1
        self.last_trade_info = None
        reward = 0.0

        act_type, direction = self.action_map[int(action)]

        # Save pre-action unrealized for shaping
        prev_ur = self._compute_unrealized_pips() if self.position != 0 else 0.0
        prev_sl = self.sl_pips

        # ---- Apply action ----
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                net, trade_sl = self._close_position(
                    "AGENT_CLOSE", self._close[self.current_step]
                )
                if trade_sl > 0:
                    reward += float(np.clip(net / trade_sl, -3.0, 3.0))

        elif act_type in ("LONG", "SHORT"):
            if self.position == 0:
                self._open_position(direction)
            # If already in position, action is ignored (no penalty, no flip)

        # ---- Advance time ----
        self.current_step += 1

        # ---- Check SL/TP on new bar ----
        result = self._check_sl_tp()
        if result is not None:
            net, trade_sl = result
            if trade_sl > 0:
                reward += float(np.clip(net / trade_sl, -3.0, 3.0))

        # ---- Shaping while holding (tiny, in R-units) ----
        if self.position != 0:
            self.time_in_trade += 1
            ur_now = self._compute_unrealized_pips()
            if prev_sl > 0:
                delta_R = (ur_now - prev_ur) / prev_sl
                reward += 0.05 * float(np.clip(delta_R, -2.0, 2.0))
            self.prev_unrealized_pips = ur_now
        else:
            self.bars_since_last_trade += 1
            # Flat reward = 0: patience is free

        # ---- Termination ----
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # ---- Ruin protection: stop if equity drops below 50% ----
        if self.equity_usd < self.initial_equity * 0.5:
            self.terminated = True

        self.equity_curve.append(float(self.equity_usd))
        obs = self._get_observation()

        # Clip total reward for PPO stability
        reward = float(np.clip(reward, -3.0, 3.0))

        info = {
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward": reward,
            "last_trade_info": self.last_trade_info,
        }

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        return obs, reward, bool(self.terminated or self.truncated), info

    def render(self):
        print(
            f"Step={self.current_step} | Eq=${self.equity_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | "
            f"SL={self.sl_price} | TP={self.tp_price}"
        )
