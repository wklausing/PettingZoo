
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from SabreEnvironment_v3 import raw_env

import supersuit as ss

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    env = raw_env(render_mode="human")
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
    model.learn(total_timesteps=2000000)
    model.save('policy')
    quit()

    sabreEnv = ss.pad_action_space_v0(raw_env(render_mode="human"))
    register_env("sabre", lambda _: PettingZooEnv(sabreEnv))

    
    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"episodes_total": 60000},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space={
            # Enviroment specific.
            "env": "sabre",
            # General
            "num_gpus": 0,
            "num_workers": 2,
            # Method specific.
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": {"main0", "main1", "main2"},
                # Always use "shared" policy.
                "policy_mapping_fn": (
                    lambda agent_id, episode, worker, **kwargs: f"main{agent_id[-1]}"
                ),
            },
        },
    ).fit()

