import functools
from pettingzoo import AECEnv
import gymnasium
from gymnasium.spaces import Discrete
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
#from typing import Tuple
import random
from gymnasium.spaces import Dict, Tuple, Discrete, Box, MultiDiscrete
from pettingzoo.test import api_test

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    api_test(env, num_cycles=1000, verbose_progress=False)
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "sabre"}

    _observation_space_cache = {}
    _action_space_cache = {}

    # Map size
    x = 10
    y = 10

    # Max number of agents and clients
    maxClients = 1
    maxCdns = 2
    maxEdgeServers = 10

    # Number of agents and clients
    numClient = 1

    observation_spaces = {}


    def __init__(self, render_mode=None, cdnCount=2):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.cdnCount = cdnCount
        self.turn_counter = 0

        self.contentProvider = ContentProvider('cp')
        self.cdns = {f'cdn{i}': CDN(id=f'cdn{i}') for i in range(cdnCount)}
        self.agentsDict = {**self.cdns, **{'cp': self.contentProvider}}
        self.agents = ['cp'] + [f'cdn{i}' for i in range(cdnCount)]#List of agent ids.
        self.possible_agents =  ['cp'] + [f'cdn{i}' for i in range(cdnCount)]#List of agent ids, which never changes.

        #TODO currently taking random cdn as fetching origin
        randomCDN = random.choice(list(self.cdns.values()))
        self.clients = {f'client{i}': Client(id=f'client{i}', position=(1,1), currentCDN=randomCDN) for i in range(self.numClient)}
        
        # optional: a mapping between agent name and agent object
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(2) for agent in self.possible_agents
        }

        self.render_mode = render_mode

        

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    def action_space(self, agent):

        if agent in self._action_space_cache:
            return self._action_space_cache[agent]

        agentObject = self.agentsDict[agent]
        action_space = None

        if agent == 'cp':
            total_edge_servers = 0
            for cdn in self.cdns.values():
                total_edge_servers += len(cdn.edgeServers)
            numClients = len(self.clients)

            action_space = Dict({
                "buyContigent": Discrete(5), # 1 Decrease by 10%, 2 Decrease by 5%, 3 No change, 4 Increase by 5%, 5 Increase by 10%
                "steerClient": Tuple([Discrete(total_edge_servers) for _ in range(numClients)]),  # Assuming 10x10 grid for positions
            })

        elif agent == 'cdn0' or agent == 'cdn1':
            action_space = Dict({
                "changePrice": Discrete(5), # 1 Decrease by 10%, 2 Decrease by 5%, 3 No change, 4 Increase by 5%, 5 Increase by 10%
                "createEdgeServer": MultiDiscrete([self.x, self.y]),  # Assuming 10x10 grid for positions
                "removeEdgeServer": Discrete(len(agentObject.edgeServers)) # Maybe the algorithm needs to have postions like above? 
            })
        else:
            raise Exception('Agent in action_space function not found.')
        self._action_space_cache[agent] = action_space
        return action_space
        
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    def observation_space(self, agent):
        print(f"observation_space({agent})")

        if agent in self._observation_space_cache:
            return self._observation_space_cache[agent]
        
        if agent == 'cp':
            # Create the observation space for Content Provider
            
            observation_space = Dict({
                'observation': Dict({
                    'clientsInfo': Dict({
                        'client0': Dict({
                            'position': Tuple([Discrete(10), Discrete(10)])
                        })
                    })
                })
            })
            self._observation_space_cache[agent] = observation_space
            self.observation_spaces[agent] = observation_space
            
            
            ## Remove later v
            # observation = {
            #     'observation': {
            #         'clientsInfo': {
            #             'client0': {
            #                 'position': np.array((1, 1), dtype=np.int32)
            #             }
            #         }
            #     }
            # }
            # observation['observation']['clientsInfo']['client0']['position'] = (1, 1)
            # if observation_space.contains(observation):
            #     print('Does it work now? HERE')
            # quit()
            ## Remove later ^
            
            return observation_space
            
        
        elif agent == 'cdn0' or agent == 'cdn1':
            # Define client positions dictionary

            clientPositions = Dict({
                'client0': Tuple([Discrete(self.x), Discrete(self.y)])
            })

            # Define CDN info dictionary
            cndData = Dict({
                f'cdn{i}': Dict(
                    {
                        'pricing': Box(low=np.array([0]), high=np.array([10]), dtype=np.float32),  # Float values, lowest is 0, highest is 10
                        'edgeServer': Tuple([Discrete(self.x), Discrete(self.y)])
                    }
                ) for i in range(self.maxCdns)
            })

            # Create the observation space
            cdnObsSpace = Dict(
                {
                    'clientPositions': clientPositions,
                    'cndData': cndData
                }
            )
        
            self._observation_space_cache[agent] = cdnObsSpace
            return cdnObsSpace
        
        else:
            raise Exception(f'Agent {agent} in observation_space function not found.')

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        print(f'observe({agent})')

        # For CP
        if agent == 'cp':

            observation = {
                'observation': {
                    'clientsInfo': {
                        'client0': {
                            'position': np.array((1, 1), dtype=np.int32)
                        }
                    }
                }
            }
            observation['observation']['clientsInfo']['client0']['position'] = (1, 1)

            self.observations[agent] = observation
            
            return observation

        # For CDN
        elif agent == 'cdn0' or agent == 'cdn1':

            observation = {
                'observation': {
                    'clientPositions': {
                        'client0': (1, 1)
                    },
                    'cndData': {
                        'cdn0': {
                            'pricing': 1.0,
                            'edgeServer': (1, 1)
                        },
                        'cdn1': {
                            'pricing': 1.0,
                            'edgeServer': (1, 1)
                        }
                    }
                }
            }
            return observation
         
        else:
            raise Exception('Agent in observe function not found.')

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        # Environment dynamics are updated after all agents have taken their turns
        self.turn_counter += 1
        if self.turn_counter >= self.num_agents:
            self.update_environment_dynamics()
            self.turn_counter = 0

        if self.render_mode == "human":
            self.render()

    def update_environment_dynamics(self):
        """
        Let environment dynamics be updated after all agents have taken their turns.
        Includes i.e. updating positions of clients and edge servers.
        """
        # for i in range(self.cdnCount):
        #     x = random.randint(0, 10)
        #     y = random.randint(0, 10)
        #     client = Client(i, (x, y), self.cdns[i])
        #     self.clients.append(client)

        for client in self.clients.values():
            client.fetchFromCdn()

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        string = "Game over"
        print(string)
        
class CDN:
    cdn = True
    edgeServers = []
    soldContigent = 0 # GBs

    def __init__(self, id, money=100, pricingFactor=1.0, initialEdgeServer=(1,1)) -> None:
        self.id = id
        self.money = money
        self.pricingFactor = pricingFactor
        self.edgeServers.append(initialEdgeServer)

    def changePrice(self, pricingFactor) -> None:
        '''
        Action: Allows CDN agent to change prices.
        '''
        self.pricingFactor = pricingFactor

    def createEdgeServer(self, position: (int, int)) -> None:
        '''
        Action: Edge servers of a CDN is defined as a list of positions.
        '''
        self.edgeServers.append(position)
        self.soldContigent -= 10

    def removeEdgeServer(self, position: (int, int)) -> None:
        '''
        Action: Remove edge server from edgeServers list.
        '''
        self.edgeServers.remove(position)

    def sellContigent(self, money) -> None:
        '''
        Action: Selling contigent of GBs to CP.
        '''
        self.money += money
        self.soldContigent += 1000


class Client:
    client = True
    reward = 0

    def __init__(self, id: str, position: (int, int), currentCDN: CDN, fetchEdgeServer=None) -> None:
        self.id = id
        self.position = position
        self.currentCDN = currentCDN
        self.fetchEdgeServer = currentCDN.edgeServers[0] if fetchEdgeServer is None else fetchEdgeServer

    def fetchFromCdn(self) -> None:
        '''
        Fetching content from CDN.
        '''
        if self.currentCDN.soldContigent > 0:
            self.currentCDN.soldContigent -= 1
            self.reward += 1
        else:
            self.reward -= 1


class ContentProvider:
    cp = True
    cdnPriceList = []

    def __init__(self, id: str, money=100) -> None:
        self.id = id
        self.money = money

    def buyContigent(self, cdn: CDN) -> None:
        '''
        Buying contigent of GBs from CDNs.
        '''
        self.money -= cdn.pricingFactor * 1000

    def steerClient(self, client: Client, cdn: CDN) -> None:
        '''
        Steering client to a CDN.
        '''
        client.fetchingOrigin = cdn


if __name__ == "__main__":

    env()
    # env = raw_env()
    # env.reset()
    # for agent in env.agent_iter():
    #     obs, reward, terminated, truncated, info = env.last()
    #     if terminated:
    #         env.step(None)
    #         print(f"Agent {agent} terminated")
    #     elif truncated:
    #         env.step(None)
    #         print("Truncated")
    #     else:
    #         action = env.action_space(agent).sample()
    #         env.step(action)
    #     #print(f"Agent: {agent}")
    #     #print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
    #     env.render()
    # env.close()