# from SabreEnvironment_v3 import raw_env

# env = raw_env(render_mode="human")
# env.reset(seed=42)

# stepCount = 0
# for agent in env.agent_iter():
#     stepCount += 1
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()

#     env.step(action)
#     if stepCount == 1000:
#         break
# env.close()



from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from SabreEnvironment_v3 import raw_env

import supersuit as ss

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO

import time

from pettingzoo.utils.env import AECEnv, ParallelEnv

def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    #env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env_fn)

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
    env_fn = raw_env

    if isinstance(raw_env, AECEnv):
        print("AEC environment detected.")
    quit()
    env_kwargs = dict(max_cycles=100, vector_state=True)
    train(env_fn, steps=81_920, seed=0)