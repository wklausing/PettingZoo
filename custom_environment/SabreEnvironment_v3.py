from pettingzoo import AECEnv
import gymnasium
from gymnasium.spaces import Discrete
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import random
from gymnasium.spaces import Dict, Discrete, Box, MultiDiscrete
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
    # env = FlattenObservation(env)
    api_test(env, num_cycles=1000, verbose_progress=False)
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "sabre"}

    _observation_space_cache = {}
    _action_space_cache = {}

    # Map size
    x = 10
    y = 10

    
    maxEdgeServers = 4 # For each CDN
    numClient = 1

    observation_spaces = {}


    def __init__(self, render_mode=None, cdnCount=2, maxEdgeServers=4, numClient=1):
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

        self.cdns = {f'cdn{i}': CDN(id=f'cdn{i}', maxEdgeServers=maxEdgeServers) for i in range(cdnCount)}
        self.contentProvider = ContentProvider(id='cp', env=self)
        self.agentsDict = {**self.cdns, **{'cp': self.contentProvider}}
        self.agents = ['cp'] + [f'cdn{i}' for i in range(cdnCount)] # List of agent ids.
        self.possible_agents =  ['cp'] + [f'cdn{i}' for i in range(cdnCount)] # List of agent ids, which never changes.

        #TODO currently taking random cdn as fetching origin
        self.clients = {f'client{i}': Client(env=self, id=f'client{i}', position=1, currentCDN=list(self.cdns.values())[0]) for i in range(self.numClient)}
        
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

        action_space = None

        if agent == 'cp':
            '''
            Possible action of CP: First, purchasing a contingent of GBs from a CDN; second, steering clients to edge server.
            '''
            total_edge_servers = sum(len(cdn.edgeServers) for cdn in self.cdns.values())
            numClients = len(self.clients)
            action_space = Dict({
                "buyContigent": Discrete(self.cdnCount + 1), # Counting up from 0 to cdnCount-1
                "steerClient": MultiDiscrete([total_edge_servers] * numClients)
            })

        elif 'cdn' in agent:
            '''
            Possible action of CDN: First, change price; second, move edge server.
            '''
            action_space = Dict({
                "changePrice": Box(low=0, high=5, shape=(1,), dtype=np.float32),
                "moveEdgeServer": Discrete(self.x * self.y) # Assuming 10x10 grid for positions
            })

        else:
            raise Exception('Agent in action_space function not found.')
        self._action_space_cache[agent] = action_space
        return action_space
        
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    def observation_space(self, agent):
        #print(f"observation_space({agent})")

        if agent in self._observation_space_cache:
            return self._observation_space_cache[agent]
        
        if agent == 'cp':
            '''
            Observation space for CP: First position of clients, second, pricing of CDNs, third, positions of edge servers.
            '''
            total_size = self.numClient + self.cdnCount + self.cdnCount * self.maxEdgeServers
            observation_space = Box(low=0, high=10, shape=(total_size,), dtype=np.float32)

        elif agent == 'cdn0' or agent == 'cdn1':
            '''
            Observation space for CDN: First position of clients, second, pricing of CDNs, third, positions of edge servers.
            '''
            total_size = self.numClient + self.cdnCount + self.cdnCount * self.maxEdgeServers
            observation_space = Box(low=0, high=10, shape=(total_size,), dtype=np.float32)
        
        else:
            raise Exception(f'Agent {agent} in observation_space function not found.')
        
        self._observation_space_cache[agent] = observation_space
        return observation_space

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        #print(f'observe({agent})')

        # For CP
        if agent == 'cp':
            '''
            Observation for CP: First position of clients, second, pricing of CDNs, third, positions of edge servers.
            '''
            client_locations = np.array([client.position for client in self.clients.values()], dtype=np.float32)
            cdn_pricing = np.array([self.agentsDict[cdn].pricingFactor for cdn in self.cdns], dtype=np.float32)
            edgeServer_locations = np.array([edgeServer for cdn in self.cdns.values() for edgeServer in cdn.edgeServers])            
            observation = np.concatenate((client_locations, cdn_pricing, edgeServer_locations))
            return observation

        # For CDN
        elif agent == 'cdn0' or agent == 'cdn1':
            '''
            Observation space for CDN: First position of clients, second, pricing of CDNs, third, positions of edge servers.
            '''
            client_locations = np.array([client.position for client in self.clients.values()], dtype=np.float32)

            cdn_pricing = np.array([self.agentsDict[cdn].pricingFactor for cdn in self.cdns], dtype=np.float32)

            edgeServer_locations = np.zeros((self.cdnCount * self.maxEdgeServers), dtype=np.float32)
            for i, cdn in enumerate(self.cdns.values()):
                for _, edgeServer in enumerate(cdn.edgeServers):
                    edgeServer_locations[i] = edgeServer

            observation = np.concatenate((client_locations, cdn_pricing, edgeServer_locations))

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

        self.contentProvider.updateCdnPriceList()

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
            # the next dead agent, or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        if agent == 'cp':
            self.doCpAction(action)
            
            for client in self.clients:
                self._cumulative_rewards[agent] += self.clients[client].reward
                self.clients[client].reward = 0
        elif 'cdn' in agent:
            self._cumulative_rewards[agent] = self.agentsDict[agent].reward
            self.agentsDict[agent].reward = 0
        else:
            pass


        
            
        # stores action of current agent
        self.state[self.agent_selection] = action

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        # Environment dynamics are updated after all agents have taken their turns
        if self._agent_selector.is_last():
            self.update_environment_dynamics()
            self.turn_counter = 0

        if self.render_mode == "human":
            self.render()

    def doCpAction(self, action):
        '''
        Doing action of CP agent.
        '''    
        if action is None: return   

        buyContigent = action['buyContigent']
        self.contentProvider.buyContigent(self.cdns['cdn' + str(buyContigent)])
        
        steerClient = action['steerClient']
        for i, j in enumerate(steerClient):
            self.contentProvider.steerClient(self.clients['client' + str(i)], self.cdns['cdn' + str(j)].edgeServers[0])

    def update_environment_dynamics(self):
        """
        Let environment dynamics be updated after all agents have taken their turns.
        Includes i.e. updating positions of clients and edge servers.
        """
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
    reward = 0

    def __init__(self, id, maxEdgeServers, money=100, pricingFactor=1.0) -> None:
        self.id = id
        self.edgeServers = [1.0] * maxEdgeServers
        self.money = money
        self.pricingFactor = pricingFactor  
        self.soldContigent = 0 # GBs   

    def changePrice(self, pricingFactor) -> None:
        '''
        Action: Allows CDN agent to change prices.
        '''
        self.pricingFactor = pricingFactor

    def moveEdgeServer(self, position: int) -> None:
        '''
        Action: Edge servers of a CDN is defined as a list of positions.
        '''
        self.edgeServers.append(position)
        self.soldContigent -= 10

    def sellContigent(self, money) -> None:
        '''
        Action: Selling contigent of GBs to CP.
        '''
        self.reward += money
        self.money += money
        self.soldContigent += money


class Client:
    reward = 0 # Later QoE
    fetchingOrigin = None

    def __init__(self, env: raw_env, id: str, position: int = 1, currentCDN: CDN = None) -> None:
        self.env = env
        self.id = id
        self.position = position
        self.currentCDN = currentCDN

    def fetchFromCdn(self) -> None:
        '''
        Fetching content from CDN.
        '''
        if self.fetchingOrigin is None:
            self.reward -= 1
        elif self.currentCDN.soldContigent > 0:
            self.currentCDN.soldContigent -= 1
            self.reward += 1
        else:
            self.reward -= 1


class ContentProvider:

    def __init__(self, env: raw_env, id: str, money=100) -> None:
        self.env = env
        self.id = id
        self.money = money
        self.cdnPriceList = [cdn.pricingFactor for cdn in env.cdns.values()]
        self.boughtContigents = [cdn.soldContigent for cdn in env.cdns.values()]

    def buyContigent(self, cdn: CDN) -> None:
        '''
        Buying contigent of GBs from CDNs.
        '''
        self.money -= 10
        index = int(cdn.id[3:])
        self.boughtContigents[index] += 10
        cdn.sellContigent(10)

    def steerClient(self, client: Client, edgeServer: float) -> None:
        '''
        Steering client to edge server.
        '''
        client.fetchingOrigin = edgeServer

    def updateCdnPriceList(self) -> None:
        '''
        Updating price list of CDNs.
        '''
        self.boughtContigents = [cdn.soldContigent for cdn in self.env.cdns.values()]

    def randomBuyAndSteer(self, cdns: dict, clients: dict) -> None:
        '''
        Randomly buying contigent from CDN and steering client to it.
        '''
        randomCDN = random.choice(list(cdns.keys()))
        self.buyContigent(cdns[randomCDN])
        cdn = cdns[randomCDN]
        self.steerClient(clients['client0'], cdn)


if __name__ == "__main__":

    env = raw_env()
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
            action = None
            if agent == 'cp':
                print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
                cpAgent = env.agentsDict[agent]
                cpAgent.updateCdnPriceList()
                cpAgent.money += reward
                if cpAgent.money <= 0:
                    print('CP agent is out of money.')
                    quit()
                elif sum(cpAgent.boughtContigents) <= 0: # Buy contigent from random CDN and steer client to it.
                    buyContigent_action = 0
                    steerClient_action = [0]
                    action = {
                        "buyContigent": buyContigent_action,
                        "steerClient": steerClient_action
                    }
                else: # Steer client to CDN with contigent available.
                    indexOfCdn = next((index for index, value in enumerate(cpAgent.boughtContigents) if value > 0), None)
                    cdn = env.cdns['cdn' + str(indexOfCdn)]
                    cpAgent.steerClient(env.clients['client0'], cdn)
                print(action)

            elif 'cdn' in agent:
                pass
            else:
                raise Exception('Agent in main function not found.')

            env.step(action)
    env.close()