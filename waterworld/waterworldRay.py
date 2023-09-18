
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    register_env("waterworld", lambda _: PettingZooEnv(waterworld_v4.env()))

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
            "env": "waterworld",
            # General
            "num_gpus": 0,
            "num_workers": 2,
            # Method specific.
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": {"main0", "main1"},
                # Always use "shared" policy.
                "policy_mapping_fn": (
                    lambda agent_id, episode, worker, **kwargs: f"main{agent_id[-1]}"
                ),
            },
        },
    ).fit()

