import numpy as np
import gym
from gym import spaces

class MindModel():
    def __init__(
        self,
        env,
        observation_space,
        action_space,
        attention_size
    ):

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.attention_size = attention_size

        input_size = self.get_input_shape(env)


    def get_input_shape(env, attention_size):
        if("mahjong" in str(env)):
            print("in mahjong environment")
            return (attention_size, 40)
    
    def predict(self, observation, mask):
        attention = self.predict_attention(observation)

        return 
    
    def predict_attention(self, observation):
        # input (6,34,4)    : input observation shape 
        # output (136)      : vector of how much something is being attended to 
        return 