import numpy as np
import torch as th
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F
from tom.utils.replay_buffer import ReplayBuffer
from tom.utils.tom import MindModel

class MATOM():
    def __init__(self, env, attention_size=10):
        env.reset()
        self.env = env 
        agents = []

        for agent_key in env.agents:
            agents.append(TOM(env=env, agent_key=agent_key, 
                            observation_space=self.env.observation_spaces[agent_key]['observation'],
                            action_space=self.env.action_spaces[agent_key],
                            attention_size=attention_size))
            
        self.agents = agents
    
    def learn(self, timesteps):
        env = self.env 

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(len(self.agents))]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        episode_step = 0
        learning_timestep = 0
        env.reset()

        while(learning_timestep < timesteps):
            learning_timestep += 1

            current_agent_index = env.agents.index(env.agent_selection)
            agent = self.agents[current_agent_index]
            player_key = env.agent_selection 
            
            obs = env.observe(agent=player_key)['observation']
            mask = env.observe(agent=player_key)['action_mask']
            action, _ = agent.predict(observation = obs, mask=mask)
            
            env.step(action)
            new_obs = env.observe(agent=player_key)['observation']

            rew_n, done_n, info_n = env.rewards, env.dones, env.infos
            player_info = info_n.get(player_key)
            done = done_n.get(player_key)
            rew_array = list(rew_n.values())
            rew = rew_array[current_agent_index]
            for agent_index, agent_reward in enumerate(rew_array):
                agent_rewards[agent_index][0] += agent_reward

            # Store data in replay buffer
            agent.replay_buffer.add(obs,new_obs,action,rew,done,[player_info])
            agent._on_step()
            #trainer.train(batch_size=trainer.batch_size, gradient_steps=trainer.gradient_steps)
            
            if(all(done_n.values())): # game is over
                final_ep_rewards.append(episode_rewards) 
                final_ep_ag_rewards.append(agent_rewards)

                episode_rewards = [0.0]  # sum of rewards for all agents
                agent_rewards = [[0.0] for _ in range(len(self.agents))]  # individual agent reward

                env.reset()
                continue

        return 

class TOM():
    def __init__(   self, 
                    env, 
                    observation_space, 
                    action_space,
                    agent_key,
                    attention_size):
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size = int(1e6), observation_space = observation_space, action_space = action_space)
        self.agent_key = agent_key
        self.agent_models = []
        self.attention_size = attention_size
    
        for tom_agent_key in env.agents:
            if(tom_agent_key != self.agent_key):
                model = MindModel(env, 
                            observation_space=env.observation_spaces[agent_key]['observation'],
                            action_space=env.action_spaces[agent_key],
                            attention_size=attention_size)

    def predict(self, observation, mask):
        # list possible actions
        # possible next states 
        # loop: next agent's possible actions 
        # get my next turn predicted value
        # roll back to estimate action values 

        

        print(observation)
        print(np.sum(mask))

        return 
   
    def __on_step():
        return 
    
