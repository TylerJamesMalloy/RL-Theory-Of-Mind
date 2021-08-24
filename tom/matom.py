import copy, math, random, gym
import numpy as np
import torch as th
from torch import nn
from gym import spaces
from torch.nn import functional as F
from tom.utils.tom import MindModel
from tom.utils.utils import get_unobserved_shape, get_concatonated_shape, get_belief_shape

class MATOM():
    def __init__(self, env, model_type="full", attention_size=20):
        env.reset()
        self.env = env 
        agents = []
        self.model_type = model_type # full, attention, belief, dqn

        for agent_key in env.agents:
            agents.append(TOM(env=env, agent_key=agent_key, 
                            observation_space=self.env.observation_spaces[agent_key]['observation'],
                            action_space=self.env.action_spaces[agent_key],
                            attention_size=attention_size,
                            model_type=self.model_type))
            
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

        prev_acts = -1 * np.ones(len(self.agents))

        while(learning_timestep < timesteps):
            learning_timestep += 1

            current_agent_index = env.agents.index(env.agent_selection)
            agent = self.agents[current_agent_index]
            player_key = env.agent_selection 

            obs = env.observe(agent=player_key)['observation']
            mask = env.observe(agent=player_key)['action_mask']

            trainers_obs = []
            trainers_reps = []
            for trainer in self.agents:
                trainer_obs = env.observe(agent=trainer.agent_key)['observation']
                trainer_obs = trainer.augment_observation(trainer_obs)
                trainers_obs.append(trainer_obs)
                trainer_rep = trainer.attention_rep(current_agent_index, trainer_obs, mask)
                trainers_reps.append(trainer_rep)

            action, _ = agent.predict(obs=obs, mask=mask)
            env.step(action)

            print("step: ", learning_timestep)

            new_obs = env.observe(agent=player_key)['observation']
            rew_n, done_n, info_n = env.rewards, env.dones, env.infos
            player_info = [info_n.get(player_key)]
            done = done_n.get(player_key)
            rew_array = list(rew_n.values())
            rew = rew_array[current_agent_index]
            for agent_index, agent_reward in enumerate(rew_array):
                agent_rewards[agent_index][0] += agent_reward
            
            for trainer in self.agents:
                prev_beliefs = trainer.prev_beliefs # needs to be called before augment observation 
                prev_act = trainer.prev_acts[trainer.agent_index]
                trainer_index = env.agents.index(trainer.agent_key)
                trainer_obs = trainers_obs[trainer_index]
                trainer_new_obs = env.observe(agent=trainer.agent_key)['observation']
                trainer_new_obs = trainer.augment_observation(trainer_new_obs)
                rep = trainers_reps[trainer_index] 
                next_rep = trainer.attention_rep(current_agent_index, trainer_new_obs, mask)

                trainer.observe(current_agent_index,trainer_obs,trainer_new_obs,rep,next_rep,action,prev_act,prev_beliefs,mask,rew,done,player_info)
                trainer._on_step()

                #trainer.train(batch_size=trainer.batch_size, gradient_steps=trainer.gradient_steps)
            
            if(all(done_n.values())): # game is over
                final_ep_rewards.append(episode_rewards) 
                final_ep_ag_rewards.append(agent_rewards)

                episode_rewards = [0.0]  # sum of rewards for all agents
                agent_rewards = [[0.0] for _ in range(len(self.agents))]  # individual agent reward

                # reset previous acts and beliefs to -1s 
                for trainer in self.agents:
                    trainer.reset()

                env.reset()
                continue

        return 

    def save(self, folder):
        for trainer in self.agents:
            trainer.save(folder + "/player_" + trainer.agent_key)


class TOM():
    def __init__(   self, 
                    env, 
                    observation_space, 
                    action_space,
                    agent_key,
                    attention_size,
                    dqn_layers = [64,64],
                    gamma = 0.999,
                    model_type="full"):
        self.env = env
        
        self.model_type = model_type
        self.agent_key = agent_key
        self.agent_index = env.agents.index(agent_key)
        self.agent_models = []
        self.attention_size = attention_size
        self.gamma = gamma
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.steps_done = 0
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.num_actions = len(env.observe(agent=agent_key)['action_mask']) + 1 # last additional action signifies no previous action
        
        self.prev_beliefs = np.zeros((len(self.env.agents), get_belief_shape(self.env)[1]))
        self.prev_acts    = self.num_actions  * np.zeros((len(self.env.agents), 1))  

        for tom_agent_index, tom_agent_key in enumerate(env.agents):
            model = MindModel(env, 
                        observation_space=get_concatonated_shape(env), #env.observation_spaces[tom_agent_key]['observation'],
                        action_space=env.action_spaces[tom_agent_key],
                        attention_size=attention_size,
                        dqn_layers=dqn_layers,
                        gamma=self.gamma,
                        device=self.device,
                        agent_index=tom_agent_index,
                        owner_index=self.agent_index,
                        model_type=self.model_type)
            self.agent_models.append(model)
        
        self.mindModel = self.agent_models[self.agent_index]
    
    def save(self, folder):
        for model_index, model in enumerate(self.agent_models):
            model.save(folder + "/minds/player_" + str(model_index))

    def attention_rep(self, trainer_index, obs, mask):
        if(self.model_type == "dqn" or self.model_type == "belief"): 
            return np.zeros(612) # no attention representation used 

        if(trainer_index == self.agent_index):
            return self.agent_models[trainer_index].attention_rep(obs, mask)
        else:
            unseen_mask = np.ones(len(mask)) # Assume other agents could perform any action
            return self.agent_models[trainer_index].attention_rep(obs, unseen_mask)

    def reset(self):
        self.prev_beliefs = np.zeros((len(self.env.agents), get_belief_shape(self.env)[1]))
        self.prev_acts    = self.num_actions * np.zeros((len(self.env.agents), 1))
        return 
        
    def masked_softmax(self, vec, mask, dim=0, epsilon=1e-12):
        exps = th.exp(vec).detach().cpu().numpy() 
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(dim) + epsilon
        return (masked_exps/masked_sums)
    
    def augment_observation(self, obs):
        prev_beliefs = []
        for agent_index, agent in enumerate(self.agent_models):
            prev_act = np.zeros(self.num_actions)  
            prev_act[int(self.prev_acts[agent_index][0])] = 1
            belief = agent.predict_beliefs(self.prev_beliefs[agent_index], prev_act)
            belief = np.reshape(belief, get_unobserved_shape(self.env)) 
            prev_beliefs.append(belief.flatten())
            obs = np.append(obs, belief, axis=0)
        self.prev_beliefs = prev_beliefs
        return obs

    def observe(self,trainer_index,obs,next_obs,rep,next_rep,action,prev_act,prev_belief,mask,reward,done,infos):
        self.prev_acts[trainer_index]  = action

        self.agent_models[trainer_index].observe(obs,next_obs,rep,next_rep,action,prev_act,prev_belief,mask,reward,done,infos)

    def predict(self, obs, mask):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample < eps_threshold:
            actions = np.linspace(0,len(mask)-1,num=len(mask), dtype=np.int32)
            return (random.choice([a for action_index, a in enumerate(actions) if mask[action_index] == 1]), None)
        else:
            rollout_steps = 0
            rollouts = 4
            obs = obs.astype('float64')
            base_copy = copy.deepcopy(self.env)
            current_copy = copy.deepcopy(self.env)
            depth_limit = 4
            current_depth = 0
            q_estimates = [[]] * len(mask)
            estimated_action = None 
            original_mask = mask
            original_agent_index = self.env.agents.index(self.env.agent_selection)

            if(self.mindModel.replay_buffer.size() < 1000):
                # Random masked action
                q_estimates = np.ones(len(original_mask))
                action_sampling = ((q_estimates * original_mask) / np.sum(q_estimates * original_mask))
                action = np.random.choice(len(original_mask), 1, p=action_sampling)[0]
                return (action, None)

            while rollout_steps < rollouts:
                current_depth += 1
                # Get current player 
                current_agent_index = self.env.agents.index(self.env.agent_selection)
                agent = self.agent_models[current_agent_index]
                # Augment state space with beliefs 
                augmented_obs = self.augment_observation(obs)
                # Predict current player q_values 
                q_mask = mask if current_agent_index == original_agent_index else np.ones(len(mask)) #  if mask shoudld be known input it, otherwise all ones
                q_values = agent.q_values(augmented_obs, q_mask) 
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
                q_mask = mask if current_agent_index == original_agent_index else np.ones(len(mask)) #  if mask shoudld be known input it, otherwise all ones 
                next_obs_q = agent.q_values(augmented_new_obs, q_mask) 
                q_estimates[estimated_action].append(reward + self.gamma * np.max(next_obs_q.detach().cpu().numpy()))

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
    
