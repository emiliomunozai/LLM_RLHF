"""Windy Highway — a 2D gridworld for visualising policy gradients.

Why this exists
---------------
Before PPO/GRPO get applied to LLMs (where the policy is opaque and the
trajectory is a sequence of tokens), it helps to *see* REINFORCE work on
a problem you can draw. This module implements the "Windy Highway" MDP
from Duane Rich's mutual_information video series, adapted to matplotlib.

The MDP
-------
- State space: continuous, the unit square `[0,1] × [0,1]`.
- Actions:     {left, right, up}.
- Dynamics:    every step the agent moves *up* by a small random delta.
               `left` / `right` shifts x by a random delta in that direction;
               `up` leaves x unchanged. A *wind* shoves the agent left or
               right depending on x (cos-shaped), so half the world is
               working *against* the agent.
- Reward:      `sin(x · 2π) + reward_shift` at each step.
               Maximum reward is at x = 0.25; minimum is at x = 0.75.
- Terminal:    when y ≥ 1.

The "high reward highway" is the vertical band near x = 0.25. A good policy
learns to drift towards it (and away from x = 0.75) despite the wind.

Policy parameterisation
-----------------------
Softmax over linear-in-features logits, where features are RBFs centred on
a hexagonal grid of "proto-points":

    f_k(s) = exp(-||s - p_k||² / σ²) / Z(s)        (k = 1..K, normalised)
    logit_a(s) = Σ_k θ_{k,a} · f_k(s)
    π(a|s) = softmax(logits(s))

`up` is fixed at logit = 0; only `left` and `right` are parameterised.
That's enough to express any meaningful behaviour on this MDP.

Training
--------
Vanilla REINFORCE:

    θ ← θ + α · G_t · ∇_θ log π(a_t | s_t)

Optionally with a value-function baseline (also RBF-features over a coarser
grid). The baseline turns the update into:

    θ ← θ + α · (G_t - V(s_t)) · ∇_θ log π(a_t | s_t)

which is unbiased but *much* lower variance — the same insight that powers
the actor-critic structure of PPO.

This module is deliberately dependency-light (numpy only) so the notebook
can focus on plotting. Two examples of how to instantiate:

    >>> hw = WindyHighway(alpha=0.5, distance_scaler=0.15, protos_per_dim=5)
    >>> hw.run(num_episodes=3000)                        # REINFORCE
    >>> hw_b = WindyHighway(alpha=0.5, distance_scaler=0.15, protos_per_dim=5,
    ...                     with_baseline=True, reward_shift=5.0)
    >>> hw_b.run(num_episodes=3000)                       # REINFORCE + baseline
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Hexagonal-grid helper (the proto-point layout)
# ---------------------------------------------------------------------------


def hexagonal_grid(n_points_x: int, n_points_y: int, max_x: float, max_y: float) -> np.ndarray:
    """Return points distributed over a hexagonal grid covering `[0,max_x] × [0,max_y]`.

    A hex layout gives smoother coverage than a square grid for the RBF
    features — every state has roughly the same distance to its nearest
    proto-point regardless of direction.
    """
    ratio = np.sqrt(3) / 2  # cos(60°)
    xv, yv = np.meshgrid(
        np.arange(n_points_x), np.arange(n_points_y), sparse=False, indexing="xy"
    )
    xv = xv * ratio
    xv[::2, :] += ratio / 2
    xv = xv * (max_x / xv.max())
    yv = yv * (max_y / yv.max())
    return np.array(
        [(xv_ij, yv_ij) for xv_i, yv_i in zip(xv, yv) for xv_ij, yv_ij in zip(xv_i, yv_i)]
    )


# ---------------------------------------------------------------------------
# The MDP + REINFORCE learner
# ---------------------------------------------------------------------------


class WindyHighway:
    """REINFORCE on the Windy Highway MDP."""

    ACTIONS: tuple[str, ...] = ("left", "right", "up")

    def __init__(
        self,
        alpha: float,
        distance_scaler: float,
        protos_per_dim: int,
        seed: int = 0,
        with_baseline: bool = False,
        reward_shift: float = 0.0,
        alpha_value: float = 0.1,
        distance_scaler_value: float = 0.15,
        protos_per_dim_value: int = 3,
        max_wind_strength: float = 0.02,
        move_min: float = 0.04,
        move_max: float = 0.06,
    ):
        self.alpha = alpha
        self.distance_scaler = distance_scaler
        self.protos_per_dim = protos_per_dim
        self.with_baseline = with_baseline
        self.reward_shift = reward_shift

        self.alpha_value = alpha_value
        self.distance_scaler_value = distance_scaler_value
        self.protos_per_dim_value = protos_per_dim_value

        self.max_wind_strength = max_wind_strength
        self.move_min_max = (move_min, move_max)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # G0s[ep] = total return of episode ep (used to draw the learning curve)
        self.G0s: list[float] = []
        self._initialise_features()

    # ---- features ---------------------------------------------------------

    def _initialise_features(self) -> None:
        self.proto_points = (
            hexagonal_grid(self.protos_per_dim, self.protos_per_dim, 0.95, 0.95) + 0.025
        )
        # θ has shape (K, 2): one row per proto-point, one column per
        # *parameterised* action (left, right). The third action (up) has a
        # fixed logit of 0.
        self.theta = np.zeros((self.proto_points.shape[0], 2))

        if self.with_baseline:
            self.proto_points_value = (
                hexagonal_grid(
                    self.protos_per_dim_value, self.protos_per_dim_value, 0.95, 0.95
                )
                + 0.025
            )
            self.w = np.zeros(self.proto_points_value.shape[0])

    def rbf_vec(self, s_xy: np.ndarray, baseline: bool = False) -> np.ndarray:
        """Normalised RBF activations of state `s_xy` against the proto-points."""
        if baseline:
            protos = self.proto_points_value
            sigma = self.distance_scaler_value
        else:
            protos = self.proto_points
            sigma = self.distance_scaler

        dist = (s_xy - protos) / sigma
        radial = np.exp(-(dist ** 2).sum(axis=1))
        return radial / radial.sum()

    # ---- environment ------------------------------------------------------

    def reward_at(self, s_xy: np.ndarray) -> float:
        """Reward of *being at* state s_xy. Maximum near x = 0.25."""
        return float(np.sin(s_xy[0] * 2 * np.pi) + self.reward_shift)

    def wind_at(self, x: float) -> float:
        """Leftward wind (positive = leftward push). cos-shaped along x."""
        return float(np.cos(x * 2 * np.pi) * self.max_wind_strength)

    def _rand_delta(self) -> float:
        return float(self.rng.uniform(*self.move_min_max))

    def get_start_state(self) -> np.ndarray:
        x = float(self.rng.uniform()) * 0.2 + 0.4   # start in [0.4, 0.6]
        return np.array([x, 0.0])

    def step(self, s_xy: np.ndarray, action: str) -> tuple[np.ndarray, float]:
        x, y = s_xy
        y = y + self._rand_delta()                     # always drift up
        if action == "left":
            x = x - self._rand_delta()
        elif action == "right":
            x = x + self._rand_delta()
        # apply wind (positive wind = leftward, since x -= wind)
        x = x - self.wind_at(s_xy[0])
        s_next = np.array([x, y]).clip(0.0, 1.0)
        return s_next, self.reward_at(s_next)

    def terminal(self, s_xy: np.ndarray) -> bool:
        return bool(s_xy[1] >= 1.0)

    # ---- policy -----------------------------------------------------------

    def logits(self, s_xy: np.ndarray) -> np.ndarray:
        rbf = self.rbf_vec(s_xy)
        left_right_logits = (self.theta * rbf[..., np.newaxis]).sum(axis=0)
        return np.concatenate([left_right_logits, np.array([0.0])])

    def action_probs(self, s_xy: np.ndarray) -> np.ndarray:
        z = np.exp(self.logits(s_xy))
        return z / z.sum()

    # ---- gradient ---------------------------------------------------------
    # ∇_θ log π(a|s) = (e_a - π(·|s)) · ∇_θ logit(s)
    # For RBF features the inner term is just rbf_vec(s), placed in the
    # column of θ that corresponds to action a (a = left or right).

    def _grad_log_prob(self, s_xy: np.ndarray, action: str) -> np.ndarray:
        rbf = self.rbf_vec(s_xy)                                # [K]
        pi = self.action_probs(s_xy)                            # [3]
        # ∂logit_a / ∂θ_{k, j}  =  rbf_k · 1{a == j}, for j in {left,right}.
        # ∂log π(a) / ∂θ = ∇logit_a − Σ_b π(b) ∇logit_b.
        grad = np.zeros_like(self.theta)                        # [K, 2]
        a_idx = self.ACTIONS.index(action)
        # Term: ∇logit_a (zero if a == up)
        if a_idx < 2:
            grad[:, a_idx] += rbf
        # Term: − Σ_b π(b) ∇logit_b
        for b_idx in range(2):
            grad[:, b_idx] -= pi[b_idx] * rbf
        return grad

    # ---- baseline ---------------------------------------------------------

    def value(self, s_xy: np.ndarray) -> float:
        if not self.with_baseline:
            return 0.0
        return float((self.w * self.rbf_vec(s_xy, baseline=True)).sum())

    # ---- one episode ------------------------------------------------------

    def play_episode(self) -> tuple[np.ndarray, list[str], list[float], list[float]]:
        s = self.get_start_state()
        states = [s]
        actions: list[str] = []
        rewards: list[float] = [float("nan")]  # rewards[0] is the "pre-action" reward (undefined)
        while not self.terminal(s):
            probs = self.action_probs(s)
            a = self.rng.choice(self.ACTIONS, p=probs)
            actions.append(a)
            s, r = self.step(s, a)
            states.append(s)
            rewards.append(r)

        # Returns: G_t = sum of future rewards (Monte-Carlo, undiscounted)
        returns: list[float] = []
        for t in range(len(rewards) - 1):
            returns.append(float(sum(rewards[t + 1 :])))
        returns.append(float("nan"))
        return np.array(states), actions, rewards, returns

    # ---- REINFORCE update -------------------------------------------------

    def update(self, states: np.ndarray, actions: list[str], returns: list[float]) -> None:
        for s, a, Gt in zip(states[:-1], actions, returns[:-1]):
            grad = self._grad_log_prob(s, a)
            if self.with_baseline:
                rbf_b = self.rbf_vec(s, baseline=True)
                v = float((self.w * rbf_b).sum())
                advantage = Gt - v
                self.w += self.alpha_value * advantage * rbf_b
                self.theta += self.alpha * advantage * grad
            else:
                self.theta += self.alpha * Gt * grad

    # ---- training loop ----------------------------------------------------

    def run(self, num_episodes: int, progress: bool = True) -> None:
        iter_ = range(num_episodes)
        if progress:
            try:
                from tqdm import tqdm
                iter_ = tqdm(iter_, desc="episodes")
            except ImportError:
                pass
        for _ in iter_:
            states, actions, _, returns = self.play_episode()
            self.G0s.append(returns[0])
            self.update(states, actions, returns)

    # ---- diagnostics ------------------------------------------------------

    def policy_field(self, n_grid: int = 12) -> dict:
        """Sample the policy on a regular grid.

        Returns a dict with the grid points and per-action probabilities,
        ready for matplotlib's quiver/streamplot.
        """
        xs = np.linspace(0.05, 0.95, n_grid)
        ys = np.linspace(0.05, 0.95, n_grid)
        Xs, Ys = np.meshgrid(xs, ys)
        left = np.empty_like(Xs)
        right = np.empty_like(Xs)
        up = np.empty_like(Xs)
        for i in range(n_grid):
            for j in range(n_grid):
                p = self.action_probs(np.array([Xs[i, j], Ys[i, j]]))
                left[i, j], right[i, j], up[i, j] = p
        return {"X": Xs, "Y": Ys, "left": left, "right": right, "up": up}

    def reward_landscape(self, n_grid: int = 200) -> dict:
        """Return the reward field over the unit square."""
        xs = np.linspace(0, 1, n_grid)
        ys = np.linspace(0, 1, n_grid)
        Xs, Ys = np.meshgrid(xs, ys)
        R = np.sin(Xs * 2 * np.pi) + self.reward_shift
        return {"X": Xs, "Y": Ys, "R": R}

    def wind_field(self, n_grid: int = 12) -> dict:
        """Return the wind direction (negative x-component) on a grid."""
        xs = np.linspace(0.05, 0.95, n_grid)
        ys = np.linspace(0.05, 0.95, n_grid)
        Xs, Ys = np.meshgrid(xs, ys)
        # wind_at returns leftward strength → quiver U = -wind
        U = -np.cos(Xs * 2 * np.pi) * self.max_wind_strength
        V = np.zeros_like(U)
        return {"X": Xs, "Y": Ys, "U": U, "V": V}

    def returns_curve(self, window: int = 50) -> dict:
        """Per-episode return + a rolling mean."""
        G = np.array(self.G0s)
        if window > 1 and len(G) >= window:
            kernel = np.ones(window) / window
            roll = np.convolve(G, kernel, mode="valid")
            roll = np.concatenate([np.full(window - 1, np.nan), roll])
        else:
            roll = G.copy()
        return {"episode": np.arange(len(G)), "G0": G, "rolling": roll}
