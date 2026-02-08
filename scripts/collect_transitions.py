import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np


def one_hot(action: int, num_actions: int) -> np.ndarray:
    v = np.zeros(num_actions, dtype=np.float32)
    v[action] = 1.0
    return v


def collect_cartpole_transitions(
    num_transitions: int,
    seed: int = 0,
    render: bool = False,
) -> dict[str, np.ndarray]:
    """
    Collect (s_t, a_t, s_{t+1}) transitions from CartPole-v1 using random actions.

    Returns a dict of numpy arrays:
      - obs:       (N, obs_dim)
      - action:    (N,) int64
      - action_oh: (N, act_dim) float32
      - next_obs:  (N, obs_dim)
      - terminated:(N,) bool
      - truncated: (N,) bool
    """
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # Seed the environment + action sampler for reproducibility
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]  # 4 for CartPole
    act_dim = env.action_space.n              # 2 for CartPole

    obs_list = []
    act_list = []
    act_oh_list = []
    next_obs_list = []
    terminated_list = []
    truncated_list = []

    # We keep stepping until we have num_transitions samples.
    while len(obs_list) < num_transitions:
        # Current state (observation) is `obs`
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Record the transition (s_t, a_t, s_{t+1})
        obs_list.append(obs.astype(np.float32))
        act_list.append(int(action))
        act_oh_list.append(one_hot(int(action), act_dim))
        next_obs_list.append(next_obs.astype(np.float32))
        terminated_list.append(bool(terminated))
        truncated_list.append(bool(truncated))

        # If the episode ended for any reason, reset to start a new episode.
        # We want a continuous stream of valid transitions.
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    env.close()

    data = {
        "obs": np.stack(obs_list, axis=0),
        "action": np.array(act_list, dtype=np.int64),
        "action_oh": np.stack(act_oh_list, axis=0),
        "next_obs": np.stack(next_obs_list, axis=0),
        "terminated": np.array(terminated_list, dtype=bool),
        "truncated": np.array(truncated_list, dtype=bool),
    }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000, help="Number of transitions to collect")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=str, default="data/cartpole_transitions.npz", help="Output .npz path")
    parser.add_argument("--render", action="store_true", help="Render the environment (slower)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = collect_cartpole_transitions(num_transitions=args.n, seed=args.seed, render=args.render)

    # Save compressed to keep file size small
    np.savez_compressed(out_path, **data)

    print(f"Saved: {out_path}")
    print("Shapes:")
    for k, v in data.items():
        print(f"  {k:10s} {v.shape}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
