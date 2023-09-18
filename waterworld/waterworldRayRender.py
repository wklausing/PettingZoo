from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4
from ray.rllib.algorithms.algorithm import Algorithm


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    register_env("waterworld", lambda _: PettingZooEnv(waterworld_v4.env()))

    my_new_ppo = Algorithm.from_checkpoint("/Users/prabu/ray_results/PPO/PPO_waterworld_5c6c9_00000_0_2023-09-18_20-08-25/checkpoint_000030")

    env = waterworld_v4.env(render_mode="human")
    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated:
            env.step(None)
            print(f"Agent {agent} terminated")
        elif truncated:
            env.step(None)
            print("Truncated")
        else:
            # Both agents use same policy, but should actually use main0 and main1.
            action = my_new_ppo.compute_single_action(obs, deterministic=True, policy_id="main0") if not terminated else None
            env.step(action)
        env.render()

