import copy 
import numpy as np
import torch as th
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F
from tom.utils.tom import MindModel
from tom.utils.utils import get_unobserved_shape, get_concatonated_shape

class MATOM():
    def __init__(self, env, attention_size=20):
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

            trainers_obs = []
            for trainer in self.agents:
                trainer_obs = env.observe(agent=trainer.agent_key)['observation']
                trainer_obs = trainer.augment_observation(trainer_obs)
                trainers_obs.append(trainer_obs)
            
            obs = env.observe(agent=player_key)['observation']
            mask = env.observe(agent=player_key)['action_mask']

            action, _ = agent.predict(obs=obs, mask=mask)
            env.step(action)

            new_obs = env.observe(agent=player_key)['observation']
            rew_n, done_n, info_n = env.rewards, env.dones, env.infos
            player_info = [info_n.get(player_key)]
            done = done_n.get(player_key)
            rew_array = list(rew_n.values())
            rew = rew_array[current_agent_index]
            for agent_index, agent_reward in enumerate(rew_array):
                agent_rewards[agent_index][0] += agent_reward

            for trainer in self.agents:
                trainer_key = env.agents.index(trainer.agent_key)
                trainer_obs = trainers_obs[trainer_key]
                trainer_new_obs = env.observe(agent=trainer.agent_key)['observation']
                trainer_new_obs = trainer.augment_observation(trainer_new_obs)

                trainer.observe(trainer_key,trainer_obs,trainer_new_obs,action,mask,rew,done,player_info)
                trainer._on_step()

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
                    attention_size,
                    dqn_layers = [64,64],
                    gamma = 1-1e-3):
        self.env = env
        
        self.agent_key = agent_key
        self.agent_index = env.agents.index(agent_key)
        self.agent_models = []
        self.attention_size = attention_size
        self.gamma = gamma

        for tom_agent_key in env.agents:
            model = MindModel(env, 
                        observation_space=get_concatonated_shape(env), #env.observation_spaces[tom_agent_key]['observation'],
                        action_space=env.action_spaces[tom_agent_key],
                        attention_size=attention_size,
                        dqn_layers=dqn_layers)
            self.agent_models.append(model)
        
        self.mindModel = self.agent_models[self.agent_index]
    
    def masked_softmax(self, vec, mask, dim=0, epsilon=1e-12):
        exps = th.exp(vec).detach().numpy() 
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(dim) + epsilon
        return (masked_exps/masked_sums)
    
    def augment_observation(self, obs):
        for agent in self.agent_models:
            belief = agent.predict_beliefs()
            belief = np.reshape(belief, get_unobserved_shape(self.env)) 
            obs = np.append(obs, belief, axis=0)
        return obs

    def observe(self,trainer_key,obs,next_obs,action,mask,reward,done,infos):
        self.agent_models[trainer_key].observe(obs,next_obs,action,mask,reward,done,infos)

    def predict(self, obs, mask):
        rollout_steps = 0
        rollouts = 10
        obs = obs.astype('float64')
        base_copy = copy.deepcopy(self.env)
        current_copy = copy.deepcopy(self.env)
        depth_limit = 8
        current_depth = 0
        q_estimates = [[]] * len(mask)
        estimated_action = None 
        original_mask = mask

        while rollout_steps < rollouts:
            current_depth += 1
            # Get current player 
            current_agent_index = self.env.agents.index(self.env.agent_selection)
            agent = self.agent_models[current_agent_index]
            # Augment state space with beliefs 
            augmented_obs = self.augment_observation(obs)
            # Predict current player q_values 
            q_values = agent.q_values(augmented_obs)
            # Get current action mask 
            mask = current_copy.observe(agent=self.env.agent_selection)['action_mask']
            # Sample action from q_values 
            action_sampling = self.masked_softmax(q_values, mask)
            if(np.sum(mask) == 0 or np.sum(action_sampling) == 0): # Game ended
                print("ERROR: Game ended erroneously")
                current_depth = 0
                current_copy = copy.deepcopy(base_copy)
                estimated_action = None
                rollout_steps += 1
                continue 
            action = np.random.choice(len(mask), 1, p=action_sampling)[0]
            # Perform action: alternatively use state transition function 
            current_copy.step(action)
            # Add reward plus discounted next observation q_value to action value estimate 
            if(estimated_action is None): estimated_action = action
            # Update state information 
            rew_n, done_n, info_n = current_copy.rewards, current_copy.dones, current_copy.infos
            reward = rew_n[self.agent_key]
            new_obs = current_copy.observe(agent=self.env.agent_selection)['observation']
            augmented_new_obs = self.augment_observation(new_obs)
            next_obs_q = agent.q_values(augmented_new_obs)
            q_estimates[estimated_action].append(reward + self.gamma * np.max(next_obs_q.detach().numpy()))

            if(all(done_n.values()) or current_depth >= depth_limit):
                current_depth = 0
                current_copy = copy.deepcopy(base_copy)
                estimated_action = None
                rollout_steps += 1
        
        q_estimates = [np.mean(i) for i in q_estimates]
        action_sampling = ((q_estimates * original_mask) / np.sum(q_estimates * original_mask))
        action = np.random.choice(len(original_mask), 1, p=action_sampling)[0]

        return (action, None)
   
    def _on_step(self):
        for trainer in self.agent_models:
            trainer.on_step()
    
