"""
src/baselines.py

Thompson Sampling and UCB1 symbolic baselines for Gaussian multi-armed bandits.

These agents are invariant to semantic arm labels — they see only arm indices
and scalar rewards. Use run_baseline() to execute N replicates and get per-turn
arrays matching the shape of LLM experiment results.

Usage
-----
from src.baselines import ThompsonSampling, UCB1, run_baseline

means  = [75.0, 50.0, 25.0]
sigma  = 12.5
T, N   = 10, 10

results = run_baseline(ThompsonSampling(k=3, sigma=sigma), means, sigma, T, N, seed=42)
results = run_baseline(UCB1(k=3), means, sigma, T, N, seed=42)

# results is a dict with arrays of shape (N, T):
#   'rewards'            per-step observed reward
#   'regrets'            per-step regret  = optimal_mean − observed_reward
#                        (matches eval_manager.py convention)
#   'cum_regrets'        cumulative regret, shape (N, T)
#   'exploration_counts' number of distinct arms pulled up to turn t, shape (N, T)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BanditAgent(ABC):
    """Common interface for symbolic bandit agents."""

    @abstractmethod
    def select_arm(self) -> int:
        """Return 0-indexed arm to pull."""
        ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Incorporate observed reward for the pulled arm."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state for a fresh replicate."""
        ...


# ---------------------------------------------------------------------------
# Thompson Sampling — Normal-Normal conjugate
# ---------------------------------------------------------------------------

class ThompsonSampling(BanditAgent):
    """
    Gaussian Thompson Sampling with a Normal-Normal conjugate model.

    Posterior per arm: mu_i | data ~ N(post_mean_i, post_var_i)

    Assumes noise variance sigma^2 is known and shared across arms.

    sigma = 0 (deterministic rewards):
        The posterior collapses to a point mass after the first observation
        of each arm, so the agent exploits perfectly once every arm has been
        seen once. This is intentional — it isolates the effect of zero noise
        in the scale-sweep conditions.

    Parameters
    ----------
    k      : number of arms
    sigma  : known noise std dev (set to 0 for deterministic rewards)
    mu0    : prior mean (default 0.0)
    sigma0 : prior std dev (default 100.0 — vague / weakly informative)
    """

    def __init__(self, k: int, sigma: float, mu0: float = 0.0, sigma0: float = 100.0):
        self.k = k
        self.sigma2 = float(sigma) ** 2
        self.mu0 = float(mu0)
        self.sigma02 = float(sigma0) ** 2
        self._post_mean: np.ndarray
        self._post_var: np.ndarray
        self.reset()

    def reset(self) -> None:
        self._post_mean = np.full(self.k, self.mu0, dtype=float)
        self._post_var = np.full(self.k, self.sigma02, dtype=float)

    def select_arm(self) -> int:
        samples = np.random.normal(self._post_mean, np.sqrt(self._post_var))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if self.sigma2 == 0.0:
            # Deterministic: observation is exact, posterior is a point mass.
            self._post_mean[arm] = reward
            self._post_var[arm] = 0.0
        else:
            # Normal-Normal conjugate update.
            v0 = self._post_var[arm]
            v_new = 1.0 / (1.0 / v0 + 1.0 / self.sigma2)
            mu_new = v_new * (self._post_mean[arm] / v0 + reward / self.sigma2)
            self._post_mean[arm] = mu_new
            self._post_var[arm] = v_new


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------

class UCB1(BanditAgent):
    """
    Standard UCB1 for multi-armed bandits.

    Index: Q_i + sqrt(2 * ln(t) / n_i)

    Each arm is force-pulled once before UCB indices are used.

    Note on scale: the confidence bonus is scale-agnostic. For your high-scale
    conditions (gap = 25), the bonus tops out at ~sqrt(2*ln(30)) ≈ 2.6 with T=30,
    so UCB1 will force-explore each arm once then exploit. For low-scale conditions
    (gap ~ 0.2), the bonus dominates and the agent explores heavily — which makes
    UCB1 a useful contrast to LLM behaviour across reward scales.
    """

    def __init__(self, k: int):
        self.k = k
        self._counts: np.ndarray
        self._values: np.ndarray
        self._t: int
        self.reset()

    def reset(self) -> None:
        self._counts = np.zeros(self.k, dtype=int)
        self._values = np.zeros(self.k, dtype=float)
        self._t = 0

    def select_arm(self) -> int:
        # Force-pull unpulled arms in order.
        unpulled = np.where(self._counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        ucb = self._values + np.sqrt(2.0 * np.log(self._t) / self._counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self._counts[arm] += 1
        self._t += 1
        # Incremental mean update.
        self._values[arm] += (reward - self._values[arm]) / self._counts[arm]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baseline(
    agent: BanditAgent,
    means: list[float],
    sigma: float,
    T: int,
    n_replicates: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run N independent replicates of a bandit episode.

    At the start of each replicate:
    - arm means are shuffled (so arm 0 does not always have the best mean)
    - agent.reset() is called (no learning carries over between replicates)

    Per-step regret = optimal_mean − observed_reward, matching the convention
    in eval_manager.py.

    Parameters
    ----------
    agent        : BanditAgent — ThompsonSampling or UCB1 instance
    means        : list of k true arm means (canonical ordering before shuffle)
    sigma        : noise std dev; 0 = deterministic
    T            : episode length (turns per replicate)
    n_replicates : number of replicates
    seed         : optional seed for reproducibility (uses numpy default_rng)

    Returns
    -------
    dict with np.ndarray of shape (n_replicates, T):
        'rewards'            per-step observed reward
        'regrets'            per-step regret = optimal_mean - observed_reward
        'cum_regrets'        cumulative regret over turns
        'exploration_counts' number of distinct arms pulled up to and including turn t
    """
    rng = np.random.default_rng(seed)
    # Also seed the legacy global state so ThompsonSampling.select_arm()
    # (which calls np.random.normal) is reproducible from the same seed.
    if seed is not None:
        np.random.seed(seed)
    means_arr = np.asarray(means, dtype=float)
    k = len(means_arr)
    optimal_mean = float(means_arr.max())

    rewards_out = np.zeros((n_replicates, T), dtype=float)
    regrets_out = np.zeros((n_replicates, T), dtype=float)
    explore_out = np.zeros((n_replicates, T), dtype=int)

    for rep in range(n_replicates):
        shuffled_means = rng.permutation(means_arr)
        agent.reset()
        seen_arms: set[int] = set()

        for t in range(T):
            arm = agent.select_arm()
            if sigma > 0:
                reward = float(rng.normal(shuffled_means[arm], sigma))
            else:
                reward = float(shuffled_means[arm])
            agent.update(arm, reward)

            rewards_out[rep, t] = reward
            regrets_out[rep, t] = optimal_mean - reward
            seen_arms.add(arm)
            explore_out[rep, t] = len(seen_arms)

    return {
        "rewards": rewards_out,
        "regrets": regrets_out,
        "cum_regrets": np.cumsum(regrets_out, axis=1),
        "exploration_counts": explore_out,
    }
