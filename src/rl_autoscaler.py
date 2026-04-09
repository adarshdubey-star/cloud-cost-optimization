"""
Reinforcement Learning-based autoscaler for cloud resource optimization.

Implements a Q-Learning agent that learns to select the optimal number of
VM instances at each time step, balancing cost against SLA violations.
The environment is a simplified cloud simulation driven by real workload data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt


class CloudEnv:
    """Gym-style environment for cloud autoscaling."""

    INSTANCE_CAPACITY_CPU = 25.0   # each VM handles ~25% of total CPU demand
    COST_PER_INSTANCE_PER_STEP = 0.05
    SLA_PENALTY_MULTIPLIER = 0.50  # penalty per % of over-utilization

    def __init__(self, cpu_series: np.ndarray, max_instances: int = 20):
        self.cpu_series = cpu_series
        self.max_instances = max_instances
        self.n_steps = len(cpu_series)
        self.t = 0
        self.current_instances = 2

    def reset(self) -> int:
        self.t = 0
        self.current_instances = 2
        return self._get_state()

    def _get_state(self) -> int:
        cpu_bucket = min(int(self.cpu_series[self.t] // 10), 9)
        inst_bucket = min(self.current_instances - 1, 9)
        return cpu_bucket * 10 + inst_bucket

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Action: 0=remove instance, 1=keep, 2=add instance."""
        if action == 0 and self.current_instances > 1:
            self.current_instances -= 1
        elif action == 2 and self.current_instances < self.max_instances:
            self.current_instances += 1

        cpu_demand = self.cpu_series[self.t]
        capacity = self.current_instances * self.INSTANCE_CAPACITY_CPU
        over_util = max(0, cpu_demand - capacity)

        cost = self.current_instances * self.COST_PER_INSTANCE_PER_STEP
        penalty = over_util * self.SLA_PENALTY_MULTIPLIER
        reward = -(cost + penalty)

        self.t += 1
        done = self.t >= self.n_steps
        next_state = self._get_state() if not done else 0
        return next_state, reward, done


class QLearningAgent:
    def __init__(self, n_states: int = 100, n_actions: int = 3,
                 lr: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.05):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_rl_autoscaler(
    cpu_data: np.ndarray,
    n_episodes: int = 150,
    output_dir: str = "results/figures",
) -> Tuple[QLearningAgent, list]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = CloudEnv(cpu_data)
    agent = QLearningAgent()

    episode_rewards = []
    episode_costs = []

    print("[RL] Training Q-Learning autoscaler...")
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_cost = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            total_cost += env.current_instances * CloudEnv.COST_PER_INSTANCE_PER_STEP
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)

        if (ep + 1) % 25 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | Reward: {total_reward:.1f} | Cost: ${total_cost:.2f} | ε: {agent.epsilon:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(episode_rewards, alpha=0.4, color="#1565C0")
    window = min(20, n_episodes // 5)
    smoothed = pd.Series(episode_rewards).rolling(window).mean()
    ax1.plot(smoothed, color="#EF5350", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("RL Agent Training: Reward Convergence")

    ax2.plot(episode_costs, alpha=0.4, color="#4CAF50")
    smoothed_cost = pd.Series(episode_costs).rolling(window).mean()
    ax2.plot(smoothed_cost, color="#FF9800", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Cost ($)")
    ax2.set_title("RL Agent Training: Cost Convergence")
    plt.tight_layout()
    fig.savefig(out / "rl_training.png", dpi=150)
    plt.close(fig)

    return agent, episode_costs


def evaluate_rl_agent(agent: QLearningAgent, cpu_data: np.ndarray) -> Dict:
    """Run the trained agent greedily and collect per-step metrics."""
    env = CloudEnv(cpu_data)
    agent.epsilon = 0.0  # greedy
    state = env.reset()

    instances_history = []
    costs = []
    sla_violations = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        instances_history.append(env.current_instances)
        step_cost = env.current_instances * CloudEnv.COST_PER_INSTANCE_PER_STEP
        costs.append(step_cost)
        capacity = env.current_instances * CloudEnv.INSTANCE_CAPACITY_CPU
        if env.t - 1 < len(cpu_data) and cpu_data[env.t - 1] > capacity:
            sla_violations += 1
        state = next_state
        if done:
            break

    total_cost = sum(costs)
    avg_instances = np.mean(instances_history)
    violation_rate = sla_violations / len(cpu_data) * 100

    return {
        "total_cost": round(total_cost, 2),
        "avg_instances": round(avg_instances, 2),
        "sla_violation_rate_pct": round(violation_rate, 2),
        "instances_history": instances_history,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/cloud_workload.csv")
    cpu = df["cpu_utilization"].values
    agent, _ = train_rl_autoscaler(cpu)
    metrics = evaluate_rl_agent(agent, cpu)
    print(f"\n[RL] Evaluation: Cost=${metrics['total_cost']}, "
          f"Avg Instances={metrics['avg_instances']}, "
          f"SLA Violations={metrics['sla_violation_rate_pct']}%")
