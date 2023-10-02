import functools
from pettingzoo import AECEnv
import gymnasium
from gymnasium.spaces import Discrete
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import random
from gymnasium.spaces import Dict, Tuple, Discrete, Box, MultiDiscrete
from pettingzoo.test import api_test


# Define the observation space
space = Dict({
    'observation': Dict({
        'a': Box(low=0, high=1, shape=(2,), dtype=np.float32),
        'b': Dict({
            'c': Discrete(2),
            'd': Box(low=0, high=1, shape=(1,), dtype=np.float32)
            })
    })
})

# Define a valid observation
observation = {
    'observation': {
        'a': np.array([0.1, 0.2], dtype=np.float32),
        'b': {
            'c': 1,
            'd': np.array([0.3], dtype=np.float32)
        }
    }
}

print(space['observation']['a'].contains(observation['observation']['a']))  # True
print(space['observation']['b']['c'].contains(observation['observation']['b']['c']))  # True
print(space['observation']['b']['d'].contains(observation['observation']['b']['d']))  # True


print('Valid observation: ', space.contains(observation)) # True
#print(observation['observation'].dtype) # Doesn't work


## Now a try with a numpy structured array ##
# Define the data type for the numpy structured array
obs_dtype = np.dtype([
    ('a', np.float32, (2,)),
    ('b', [
        ('c', np.int32),
        ('d', np.float32, (1,))
    ])
])
observation_np = np.zeros(1, dtype=obs_dtype)

# Populate the numpy structured array
observation_np['a'] = np.array([0.1, 0.2], dtype=np.float32)
observation_np['b']['c'] = 1
observation_np['b']['d'] = np.array([0.3], dtype=np.float32)

# Create the final observation dictionary
observation = {'observation': observation_np}

print(space['observation']['a'].contains(observation['observation']['a']))  # False
print(space['observation']['b']['c'].contains(observation['observation']['b']['c']))  # False
print(space['observation']['b']['d'].contains(observation['observation']['b']['d']))  # False

# Check if the observation is valid
print('Valid observation:', space.contains(observation)) # False
print(observation['observation'].dtype) # Does work

## api_test.py ##
# Line 383
# assert env.observation_space(agent).contains(
#     prev_observe
# ), "Out of bounds observation: " + str(prev_observe)

# if isinstance(env.observation_space(agent), gymnasium.spaces.Box):
#     assert env.observation_space(agent).dtype == prev_observe.dtype
# elif isinstance(env.observation_space(agent), gymnasium.spaces.Dict):
#     assert (
#         env.observation_space(agent)["observation"].dtype
#         == prev_observe["observation"].dtype
#     )
# Line 395