
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.butterfly import knights_archers_zombies_v10

import supersuit as ss

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO

from custom_environment.sabre_v1 import env, raw_env
from pettingzoo import AECEnv

import time

from pettingzoo.utils.conversions import aec_to_parallel

from supersuit import pad_action_space_v0

def train(env, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    #env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    #env = ss.black_death_v3(env_fn)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Use a CNN policy if the observation space is visual
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

if __name__ == "__main__":
    
    env = env()
    env = pad_action_space_v0(env)
    env = aec_to_parallel(env)
    
    train(env, steps=81_920, seed=0)