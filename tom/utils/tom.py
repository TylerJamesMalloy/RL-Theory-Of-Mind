from os import stat
import numpy as np
import math 
import gym
from gym import spaces
from numpy.lib.utils import info
from torch._C import device
from tom.utils.replay_buffer import ReplayBuffer
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        attention_size = 20,
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
        gamma
    ):

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.attention_size = attention_size
        self.device = device
        self.gamma = gamma

        # move this to util settings object
        self.batch_size = 16
        self.target_update = 10

        input_size = get_input_shape(env=env, attention_size=attention_size)

        buffer_obs_space = spaces.Box(0,1, shape=observation_space, dtype=np.int32)
        buffer_rep_space = spaces.Box(0,1, shape=(input_size,), dtype=np.int32)
        self.replay_buffer = ReplayBuffer(  buffer_size=int(1e5), 
                                            observation_space=buffer_obs_space, 
                                            action_space=action_space,
                                            representation_space=buffer_rep_space) # can't commit too much memory? 

        self.obs_shape = math.prod(list(observation_space))
        self.policy_net = DQN(input_shape=input_size, output_shape=action_space.n, layers=dqn_layers, device=self.device)
        self.target_net = DQN(input_shape=input_size, output_shape=action_space.n, layers=dqn_layers, device=self.device)
        self.optimizer = optim.RMSprop(self.policy_net.model.parameters())

        attention_input = self.obs_shape + action_space.n
        attention_output = get_attention_shape(env=env)
        self.attention = AttentionModel(input_shape=attention_input, output_shape=attention_output, layers=dqn_layers, attention_size=attention_size, device=self.device)

        self.belief_shape, self.belief_output = get_belief_shape(env=env)
        belief_input = sum(list(self.belief_shape))
        self.belief = BeliefModel(input_shape=belief_input, output_shape=self.belief_output, layers=dqn_layers, device=self.device)

        self.unobserved_shape = get_unobserved_shape(env=env)
        self.prev_belief = np.zeros(self.belief_shape[0]) # size of belief space 
        self.prev_action = np.zeros(self.belief_shape[1]) # size of action space

        self.training_step = 0
    
    def attention_rep(self, obs, mask):
        attention = self.predict_attention(obs, mask).data.cpu().numpy()
        top_n_idx = np.argsort(attention)[-self.attention_size:]    # could alter to be 'soft' 
        return get_compressed_obs(env=self.env, obs=obs, attention_idxs=top_n_idx)

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
    
    def observe(self,obs,next_obs,rep,next_rep,action,mask,reward,done,infos):
        self.replay_buffer.add(obs,next_obs,rep,next_rep,action,mask,reward,done,infos)
    
    def predict_beliefs(self): 
        input = np.concatenate((self.prev_belief.flatten() , self.prev_action.flatten())).flatten()
        input_tensor = th.from_numpy(input).to(self.device)
        belief = self.belief.predict(input_tensor)
        belief = belief.data.cpu().numpy()
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
        (obs, next_obs, reps, next_reps, acts, masks, dones, rewards) = transitions

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = th.tensor(tuple(map(lambda s: s is not None, next_obs)), device=self.device, dtype=th.bool)
        non_final_next_states = th.stack([s for s in next_reps if s is not None])
        state_batch = obs #th.cat(obs)
        action_batch = acts #th.cat(acts)
        reward_batch = rewards #th.cat(rewards)
        reps_batch = reps
        next_reps_batch = next_reps
        reward_batch = reward_batch.squeeze()
        mask_batch = masks
        reps_batch = th.flatten(reps_batch, start_dim=1)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net.predict(reps_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = th.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net.predict(non_final_next_states).max(1).values.detach() # max should be calculated w.r.t next state mask? 
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # TODO: check why loss target is sometimes (16,1) instead of (16,0)
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.training_step % self.target_update == 0: # Set target net to policy net
            self.target_net.model.load_state_dict(self.policy_net.model.state_dict())

        ##### train attention net ######
        attention_input = th.cat((state_batch.flatten(start_dim=1), mask_batch), 1)
        attentions = self.attention.model(attention_input.float())




        assert(False)
        
        #####  train belief net   ######

