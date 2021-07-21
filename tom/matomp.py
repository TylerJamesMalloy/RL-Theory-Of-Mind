import numpy as np
import torch as th
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F

class MATOMP():
    def __init__(self, env):
        self.env = env 
        agents = []

        for agent_key in env.agents:
            setattr(env, "action_space", env.action_spaces[agent_key])
            setattr(env, "observation_space", env.observation_spaces[agent_key]['observation'])

            if(hasattr(env.unwrapped, 'reward_range')):
                setattr(env, "reward_range", env.unwrapped.reward_range)
            else:
                setattr(env, "reward_range", (0,1))

            agents.append(TOMP("MlpPolicy", env))
            
        self.agents = agents




class TOMP():
    def __init__(self, env):
        self.env = env