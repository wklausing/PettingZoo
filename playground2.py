
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.butterfly import knights_archers_zombies_v10

import supersuit as ss

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO

from custom_environment import sabre_v1
from pettingzoo import AECEnv

import time

from pettingzoo.utils.conversions import aec_to_parallel

from supersuit import pad_action_space_v0

import glob
import os

def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    # env = sabre_v1.raw_env(render_mode=render_mode, **env_kwargs)
    
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        #env.action_space(env.possible_agents[0]).seed(i)

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        for agent in env.agents:
            rewards[agent] += env.rewards[agent]

        if termination or truncation:
            break
        else:
            if 'cdn_1' in agent:
                print('Here')

            if 'cp' in agent:
                print('Here')
            if agent == env.possible_agents[0]:
                act = env.action_space(agent).sample()
            else:
                act = model.predict(obs, deterministic=True)[0]

        print(act)
        env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward

if __name__ == "__main__":
    
    foo = knights_archers_zombies_v10.env()
