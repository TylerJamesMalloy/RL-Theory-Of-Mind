from copy import deepcopy
from os import stat
import numpy as np
import math 
import gym
from gym import spaces
from numpy.core.numeric import indices
from numpy.lib.utils import info
from torch._C import device
from tom.utils.replay_buffer import ReplayBuffer
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tom.utils.utils import * #get_attention_shape, get_compressed_obs, get_input_shape, get_belief_shape, get_unobserved_shape

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
        layers = [64,64],
        device='cuda'
    ):
        self.model = MLP(input_shape = input_shape, output_shape = output_shape, layers = layers).to(device)

    def predict(self, obs):
        return self.model(obs.float())

class AttentionModel():
    def __init__(
        self, 
        input_shape, 
        output_shape, 
        layers = [64,64],
        attention_size = 10,
        device='cuda'
    ):
        self.attention_size = attention_size
        self.model = MLP(input_shape = input_shape, output_shape = output_shape, layers = layers).to(device)

    def predict(self, obs):
        return self.model(obs.float()) 

class BeliefModel():
    def __init__(
        self, 
        input_shape, 
        output_shape, 
        layers = [64,64],
        device='cuda'
    ):
        self.model = MLP(input_shape = input_shape, output_shape = output_shape, layers = layers).to(device)

    def predict(self, obs):
        return self.model(obs.float()) 

class MindModel():
    def __init__(
        self,
        env,
        observation_space,
        action_space,
        attention_size,
        dqn_layers,
        device,
        gamma,
        agent_index,
        owner_index
    ):

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.attention_size = attention_size
        self.device = device
        self.gamma = gamma
        self.agent_index = agent_index
        self.owner_index = owner_index

        # move this to util settings object
        self.batch_size = 256
        self.target_update = 10

        input_size = get_input_shape(env=env, attention_size=attention_size)
        belief_shape = (len(self.env.agents), get_belief_shape(env=env)[1])

        buffer_obs_space = spaces.Box(0,1, shape=observation_space, dtype=np.int32)
        buffer_rep_space = spaces.Box(0,1, shape=(input_size,), dtype=np.int32)
        self.replay_buffer = ReplayBuffer(  buffer_size=int(1e5), 
                                            observation_space=buffer_obs_space, 
                                            action_space=action_space,
                                            representation_space=buffer_rep_space,
                                            belief_shape=belief_shape) # can't commit too much memory? 

        self.obs_shape = math.prod(list(observation_space))
        self.policy_net = DQN(input_shape=input_size, output_shape=action_space.n, layers=dqn_layers, device=self.device)
        self.target_net = DQN(input_shape=input_size, output_shape=action_space.n, layers=dqn_layers, device=self.device)
        self.policy_optimizer = optim.RMSprop(self.policy_net.model.parameters())

        attention_input = self.obs_shape + action_space.n
        attention_output = get_attention_shape(env=env)
        self.attention = AttentionModel(input_shape=attention_input, output_shape=attention_output, layers=dqn_layers, attention_size=attention_size, device=self.device)
        self.attention_optimizer = optim.RMSprop(self.attention.model.parameters())

        self.belief_shape, self.belief_output = get_belief_shape(env=env)
        belief_input = sum(list(self.belief_shape))
        self.belief = BeliefModel(input_shape=belief_input, output_shape=self.belief_output, layers=dqn_layers, device=self.device)
        self.belief_optimizer = optim.RMSprop(self.belief.model.parameters())
        
        self.all_parameters = list(self.policy_net.model.parameters()) + list(self.attention.model.parameters()) + list(self.belief.model.parameters())
        self.optimizer = optim.RMSprop(self.all_parameters)

        self.unobserved_shape = get_unobserved_shape(env=env)
        self.prev_belief = np.zeros(self.belief_shape[0]) # size of belief space 
        self.prev_action = np.zeros(self.belief_shape[1]) # size of action space

        self.training_step = 0
    
    def attention_rep(self, obs, mask):
        attention = self.predict_attention(obs, mask).data.cpu().numpy()
        top_n_idx = np.argsort(attention)[-self.attention_size:]  
        top_n_val = attention[top_n_idx]  
        return get_compressed_obs(env=self.env, obs=obs, attention_idxs=top_n_idx, attention_vals=top_n_val)

    def q_values(self, obs, mask):
        compressed_obs = self.attention_rep(obs, mask)
        return self.policy_net.predict(th.from_numpy(compressed_obs).to(self.device))

    def predict(self, obs, mask):
        # copy of functionality in matom model 
        q_values = self.q_values(obs, mask)
        exps = th.exp(q_values).detach().cpu().numpy() 
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(0) + 1e-12
        action_sampling = (masked_exps/masked_sums)
        action = np.random.choice(len(mask), 1, p=action_sampling)[0]

        return action
    
    def observe(self,obs,next_obs,rep,next_rep,action,prev_act,prev_belief,mask,reward,done,infos):
        self.replay_buffer.add(obs,next_obs,rep,next_rep,action,prev_act,prev_belief,mask,reward,done,infos)
    
    def predict_beliefs(self, prev_belief, prev_action):
        prev_belief = th.from_numpy(prev_belief).flatten()
        prev_action = th.from_numpy(prev_action)
        input = th.cat((prev_belief, prev_action)).to(device=self.device)
        belief = self.belief.predict(input)
        belief = belief.data.cpu().numpy() # should this be tensor or numpy 
        # reshape to shape of unobnserved
        belief = np.reshape(belief, self.unobserved_shape)

        return belief
    
    def save(self, folder):
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        th.save(self.policy_net.model.state_dict(), folder + "/policy")
        th.save(self.target_net.model.state_dict(), folder + "/target")
        th.save(self.belief.model.state_dict(), folder + "/belief")
        th.save(self.attention.model.state_dict(), folder + "/attention")

    def predict_attention(self, obs, mask):
        # Example for mahjong environment 
        # input (10,34,4)   : input obs shape + action 
        # output (136)      : vector of how much something is being attended to 
        attention_in = np.concatenate((obs.flatten(), mask.astype(np.float)))
        attention_in = th.from_numpy(attention_in).to(self.device)
        return self.attention.predict(attention_in)

    def on_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        self.training_step += 1

        transitions = self.replay_buffer.sample(self.batch_size, self.env)
        # Transpose the batch
        #batch = Transition(*zip(*transitions))
        (obs, next_obs, _, _, acts, prev_acts, prev_beliefs, masks, dones, rewards) = transitions

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = th.tensor(tuple(map(lambda s: s is not None, next_obs)), device=self.device, dtype=th.bool) 
        non_final_next_obs = np.asarray([s for s in next_obs.cpu().numpy() if s is not None])
        non_final_next_obs = th.from_numpy(non_final_next_obs).to(device=self.device) 
        rewards = rewards.squeeze() # rewards may be in a 1D vector 

        ##### belief augmentation ######
        prev_acts = F.one_hot(prev_acts.squeeze(), num_classes=39) # ohe previous action 
        belief_input = th.cat((prev_beliefs, prev_acts), dim=1)
        belief = self.belief.predict(belief_input)
        belief = th.reshape(belief, (16, 34, 4)) # my beliefs 
        obs[:, 6 + self.agent_index, :, :] = belief # set belief to my belief, this didn't add the belief prediction to the graph 

        belief = belief.unsqueeze(dim=1)
        obs = th.cat((obs[:, 0:6 + self.agent_index, :, :], belief, obs[:, 7 + self.agent_index:10, :, :]), dim=1)

        ##### get attention ######
        attention_input = th.cat((obs.flatten(start_dim=1), masks), 1)
        attentions = self.attention.model(attention_input.float()) # Get index of top K attentions 
        compressed = get_compressed(attentions, obs, self.attention_size, self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net.model(compressed).gather(dim=1, index=acts)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = th.zeros(self.batch_size, device=self.device)

        # TODO: add next belief state prediction ?? 
        # Currently it is only the one from memory 
        next_masks = th.ones_like(masks, device=self.device) # next mask is unknown 
        next_attention_input = th.cat((non_final_next_obs.flatten(start_dim=1), next_masks), 1)
        next_attentions = self.attention.model(next_attention_input.float())
        next_compressed = get_compressed(next_attentions, non_final_next_obs, self.attention_size, self.device) # This is all torch operations so it remains on the graph 

        next_state_values[non_final_mask] = self.target_net.predict(next_compressed).max(1).values.detach() 
        # Compute the expected Q values
        # If we are modelling ourselves, we update according to observed reward 
        if(self.owner_index == self.agent_index):
            expected_state_action_values = (next_state_values * self.gamma) + rewards # predicting reward 
        # Otherwise we set the reward of the observed action to 1 if we predicted it and otherwise to zero 
        else:
            predicted_action = state_action_values.max(1).values.detach() 
            rewards = predicted_action.eq(acts.squeeze()) # turn into next state value
            expected_state_action_values = (next_state_values * self.gamma) + rewards # predicting action correct, opponent reward not observed 
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.all_parameters:
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.training_step % self.target_update == 0: # Set target net to policy net
            self.target_net.model.load_state_dict(self.policy_net.model.state_dict())
        
        

