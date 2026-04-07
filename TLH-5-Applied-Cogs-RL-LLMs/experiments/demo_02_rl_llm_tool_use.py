"""
demo_02_rl_llm_tool_use.py
==========================
TLH-1: Applied Cognitive Science × RL × LLMs

Goal
----
Train a lightweight RL policy (PPO via TRL / Stable-Baselines3) on a
synthetic tool-selection environment. Compare against two baselines:
  1. Random tool selection
  2. Greedy / ReAct-style rule-based selection

The environment simulates an enterprise query-answering scenario where the
agent must choose the right tool (web search, database query, code execution,
calendar lookup, escalate) to answer each query.

Usage
-----
    python demo_02_rl_llm_tool_use.py                 # train + evaluate
    python demo_02_rl_llm_tool_use.py --eval-only     # evaluate saved model
    python demo_02_rl_llm_tool_use.py --episodes 500  # custom episode count

Dependencies
------------
    pip install stable-baselines3 gymnasium sentence-transformers numpy matplotlib
"""

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[WARNING] stable-baselines3 not found — using mock RL agent")

import matplotlib.pyplot as plt


# ── Tool Definitions ──────────────────────────────────────────────────────────

TOOLS = [
    "web_search",
    "database_query",
    "code_execution",
    "calendar_lookup",
    "escalate_to_human",
]

# Map each query type to the optimal tool
OPTIMAL_TOOL: Dict[str, str] = {
    "factual_question":    "web_search",
    "data_retrieval":      "database_query",
    "code_generation":     "code_execution",
    "scheduling":          "calendar_lookup",
    "complex_regulatory":  "escalate_to_human",
}

QUERY_TYPES = list(OPTIMAL_TOOL.keys())


# ── Custom Gymnasium Environment ──────────────────────────────────────────────

@dataclass
class Query:
    query_type: str
    text: str
    optimal_tool: str


def _sample_query() -> Query:
    qt = random.choice(QUERY_TYPES)
    templates = {
        "factual_question":   "What is the market cap of {company}?",
        "data_retrieval":     "Fetch the last 30 days of {metric} data from the warehouse.",
        "code_generation":    "Write a Python function to {task}.",
        "scheduling":         "Book a meeting with {name} next Tuesday at 3pm.",
        "complex_regulatory": "Assess our GDPR exposure for the new {product} launch.",
    }
    companies  = ["Apple", "Google", "Microsoft", "Amazon"]
    metrics    = ["revenue", "churn", "DAU", "NPS"]
    tasks      = ["parse JSON", "sort a list", "call an API"]
    names      = ["Alice", "Bob", "Carol"]
    products   = ["AI feature", "data pipeline", "mobile app"]

    text = (
        templates[qt]
        .replace("{company}", random.choice(companies))
        .replace("{metric}",  random.choice(metrics))
        .replace("{task}",    random.choice(tasks))
        .replace("{name}",    random.choice(names))
        .replace("{product}", random.choice(products))
    )
    return Query(query_type=qt, text=text, optimal_tool=OPTIMAL_TOOL[qt])


class ToolSelectionEnv(gym.Env):
    """
    A simple environment where the agent observes a query type (as a one-hot
    vector) and must select the correct tool.

    Observation: one-hot vector over query types  [n_query_types]
    Action:      discrete tool index              [n_tools]
    Reward:
        +1.0  correct tool selected
        -0.5  wrong tool selected
        -1.0  escalated when not needed (conservative penalty)
        episode terminates after each query (episodic MDP)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.n_query_types = len(QUERY_TYPES)
        self.n_tools        = len(TOOLS)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_query_types,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_tools)

        self._current_query: Optional[Query] = None

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.n_query_types, dtype=np.float32)
        idx = QUERY_TYPES.index(self._current_query.query_type)
        obs[idx] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_query = _sample_query()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        selected_tool = TOOLS[action]
        optimal_tool  = self._current_query.optimal_tool

        if selected_tool == optimal_tool:
            reward = 1.0
        elif selected_tool == "escalate_to_human" and optimal_tool != "escalate_to_human":
            reward = -1.0   # unnecessary escalation is costly
        else:
            reward = -0.5

        terminated = True   # one query per episode
        truncated  = False
        info = {
            "query_type":    self._current_query.query_type,
            "selected_tool": selected_tool,
            "optimal_tool":  optimal_tool,
            "correct":       selected_tool == optimal_tool,
        }
        self._current_query = _sample_query()  # pre-load next
        return self._get_obs(), reward, terminated, truncated, info


# ── Baseline Agents ───────────────────────────────────────────────────────────

class RandomAgent:
    """Selects a tool uniformly at random."""
    def predict(self, obs, deterministic=False):
        return random.randint(0, len(TOOLS) - 1), None


class GreedyReActAgent:
    """
    Rule-based agent that maps query-type heuristics to tools.
    Mimics a typical ReAct-style prompt-engineered system.
    """
    _rules = {
        0: 0,  # factual_question   → web_search
        1: 1,  # data_retrieval     → database_query
        2: 2,  # code_generation    → code_execution
        3: 3,  # scheduling         → calendar_lookup
        4: 4,  # complex_regulatory → escalate_to_human
    }

    def predict(self, obs, deterministic=False):
        qt_idx = int(np.argmax(obs))
        return self._rules.get(qt_idx, 0), None


# ── Evaluation Utility ────────────────────────────────────────────────────────

def evaluate_agent(agent, env: ToolSelectionEnv, n_episodes: int = 200) -> Dict:
    successes = 0
    rewards   = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action, _ = agent.predict(obs)
        _, reward, _, _, info = env.step(int(action))
        successes += info["correct"]
        rewards.append(reward)

    return {
        "success_rate":  successes / n_episodes,
        "mean_reward":   float(np.mean(rewards)),
        "std_reward":    float(np.std(rewards)),
        "n_episodes":    n_episodes,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 20_000) -> "PPO":
    """Train a PPO agent on the ToolSelectionEnv."""
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 is required for RL training")

    vec_env = make_vec_env(ToolSelectionEnv, n_envs=4)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,   # entropy bonus encourages exploration
        tensorboard_log="./tb_logs/",
    )
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("Training complete.")
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results: Dict[str, Dict]) -> None:
    agents = list(results.keys())
    success_rates = [results[a]["success_rate"] * 100 for a in agents]
    mean_rewards  = [results[a]["mean_reward"]          for a in agents]

    colours = ["#E74C3C", "#F39C12", "#2ECC71"][:len(agents)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Success rate
    bars = axes[0].bar(agents, success_rates, color=colours, edgecolor="white", linewidth=1.2)
    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0].set_title("Tool Selection — Success Rate", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 110)
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    # Mean reward
    bars2 = axes[1].bar(agents, mean_rewards, color=colours, edgecolor="white", linewidth=1.2)
    axes[1].set_ylabel("Mean Reward", fontsize=12)
    axes[1].set_title("Tool Selection — Mean Reward", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, mean_rewards):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.02 if val >= 0 else -0.08),
                     f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("tool_selection_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to tool_selection_comparison.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    env = ToolSelectionEnv()

    results: Dict[str, Dict] = {}

    # Baseline 1: Random
    print("\n[1/3] Evaluating Random agent...")
    results["Random"] = evaluate_agent(RandomAgent(), env, args.eval_episodes)

    # Baseline 2: Greedy/ReAct
    print("[2/3] Evaluating Greedy/ReAct agent...")
    results["Greedy (ReAct)"] = evaluate_agent(GreedyReActAgent(), env, args.eval_episodes)

    # RL agent
    if not args.eval_only and SB3_AVAILABLE:
        print("[3/3] Training & evaluating PPO (RL) agent...")
        model = train_ppo(total_timesteps=args.episodes * 10)
        results["PPO (RL)"] = evaluate_agent(model, env, args.eval_episodes)
    elif SB3_AVAILABLE:
        print("[3/3] Skipping training (--eval-only). Using untrained PPO...")
        vec_env = make_vec_env(ToolSelectionEnv, n_envs=1)
        model = PPO("MlpPolicy", vec_env, verbose=0)
        results["PPO (untrained)"] = evaluate_agent(model, env, args.eval_episodes)
    else:
        print("[3/3] stable-baselines3 unavailable — skipping RL agent")

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Agent':<22} {'Success Rate':>14} {'Mean Reward':>13}")
    print("-" * 60)
    for agent_name, res in results.items():
        print(f"{agent_name:<22} {res['success_rate']*100:>13.1f}% {res['mean_reward']:>13.3f}")
    print("=" * 60)

    # Plot
    plot_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL vs Baseline Tool-Use Demo")
    parser.add_argument("--episodes",       type=int, default=2000,
                        help="RL training episodes (default: 2000)")
    parser.add_argument("--eval-episodes",  type=int, default=500,
                        help="Evaluation episodes per agent (default: 500)")
    parser.add_argument("--eval-only",      action="store_true",
                        help="Skip training; evaluate agents only")
    args = parser.parse_args()
    main(args)
