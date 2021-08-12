import numpy as np
import math 
import gym
from gym import spaces
from numpy.lib.utils import info
from tom.utils.replay_buffer import ReplayBuffer
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tom.utils.utils import get_attention_shape, get_compressed_obs, get_input_shape, get_belief_shape, get_unobserved_shape

class MLP(nn.Module):
    def __init__(self,
        input_shape,
        output_shape,
        layers):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(input_shape, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_shape)
        
    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = th.sigmoid(self.fc3(x))
        return x

class DQN():
    def __init__(
        self, 
        input_shape, 
        output_shape, 
        layers = [64,64]
    ):
        self.model = MLP(input_shape = input_shape, output_shape = output_shape, layers = layers)

    def predict(self, obs):
        return self.model(obs.float())

class AttentionModel():
    def __init__(
        self, 
        input_shape, 
        output_shape, 
        layers = [64,64],
        attention_size = 20,
    ):
        self.attention_size = attention_size
        self.model = MLP(input_shape = input_shape, output_shape = output_shape, layers = layers)

    def predict(self, obs):
        return self.model(obs.float()) 

class BeliefModel():
    def __init__(
        self, 
        input_shape, 
        output_shape, 
        layers = [64,64],
    ):
        self.model = MLP(input_shape = input_shape, output_shape = output_shape, layers = layers)

    def predict(self, obs):
        return self.model(obs.float()) 

class MindModel():
    def __init__(
        self,
        env,
        observation_space,
        action_space,
        attention_size,
        dqn_layers
    ):

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.attention_size = attention_size

        buffer_obs_space = spaces.Box(0,1, shape=observation_space, dtype=np.int32)
        self.replay_buffer = ReplayBuffer(buffer_size = int(1e5), observation_space=buffer_obs_space, action_space=action_space) # can't commit too much memory? 

        self.obs_shape = math.prod(list(observation_space)) # needs to include unobserved space 
        input_size = get_input_shape(env=env, attention_size=attention_size)
        self.dqn = DQN(input_shape = input_size, output_shape = action_space.n, layers=dqn_layers)

        attention_output = get_attention_shape(env=env)
        self.attention = AttentionModel(input_shape = self.obs_shape, output_shape = attention_output, layers=dqn_layers, attention_size=attention_size)

        self.belief_shape, self.belief_output = get_belief_shape(env=env)
        belief_input = sum(list(self.belief_shape))
        self.belief = BeliefModel(input_shape=belief_input, output_shape=self.belief_output, layers=dqn_layers)

        self.unobserved_shape = get_unobserved_shape(env=env)
        self.prev_belief = np.zeros(self.belief_shape[0]) # size of belief space 
        self.prev_action = np.zeros(self.belief_shape[1]) # size of action space

    def on_step(self):
        # train attention model from memory 
        print("training mind model")
        # train belief model from memory 
        # train dqn model from memory 
        assert(False)
        return 
    
    def q_values(self, obs, mask=None):
        attention = self.predict_attention(obs).data.numpy()
        top_n_idx = np.argsort(attention)[-self.attention_size:]    # could alter to be 'soft' 
        compressed_obs = get_compressed_obs(env=self.env, obs=obs, attention_idxs=top_n_idx)

        return self.dqn.predict(th.from_numpy(compressed_obs))

    def predict(self, obs, mask):
        # copy of functionality in matom model 
        q_values = self.q_values(obs, mask)
        exps = th.exp(q_values).detach().numpy() 
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(0) + 1e-12
        action_sampling = (masked_exps/masked_sums)
        action = np.random.choice(len(mask), 1, p=action_sampling)[0]

        return action
    
    def observe(self,obs,next_obs,action,mask,reward,done,infos):
        self.replay_buffer.add(obs,next_obs,action,mask,reward,done,infos)
    
    def predict_beliefs(self): 
        input = np.concatenate((self.prev_belief.flatten() , self.prev_action.flatten())).flatten()
        input_tensor = th.from_numpy(input)
        belief = self.belief.predict(input_tensor)
        belief = belief.data.numpy()
        # reshape to shape of unobnserved
        belief = np.reshape(belief, self.unobserved_shape)

        return belief

    def predict_attention(self, obs):
        # Example for mahjong environment 
        # input (10,34,4)   : input obs shape 
        # output (136)      : vector of how much something is being attended to 
        obs_tensor = th.from_numpy(obs.flatten())
        return self.attention.predict(obs_tensor)

