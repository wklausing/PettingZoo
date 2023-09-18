from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.butterfly import pistonball_v6
from ray.rllib.algorithms.algorithm import Algorithm


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    env_name = "pistonball_v6"

    register_env("pistonball_v6", lambda _: PettingZooEnv(pistonball_v6.env()))

    register_env(env_name, lambda _: PettingZooEnv(pistonball_v6.env()))

    my_new_ppo = Algorithm.from_checkpoint("/Users/prabu/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_ed2d2_00000_0_2023-09-03_16-12-33/params.pkl")

    env = pistonball_v6.env(render_mode="human")
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
            action = my_new_ppo.compute_single_action(obs, policy_id=agent)
            env.step(action)
        env.render()

