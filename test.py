from pettingzoo.tom import liars_dice_v0 
from pettingzoo.classic import backgammon_v3
from tom.madqn import MADQN, DQN
import numpy as np
import time 

#env = liars_dice_v0.env(num_players=2, num_dice=5, num_sides=6, common_hand=False, opponents_hand_visible=False)
env = backgammon_v3.env()

a0 = "random" #= DQN.load("backgammon/100K/0.zip")
a1 = DQN.load("backgammon/10M/1.zip")
agents = [a0,a1]
final_ep_ag_rewards = [0, 0]

timesteps = 100000
ts = 0
episode_step = 0
episodes = 0 

obs = env.reset()
while ts < timesteps:
    current_agent_index = env.agents.index(env.agent_selection)
    episode_step += 1
    trainer = agents[current_agent_index]
    player_key = env.agents[current_agent_index]

    obs = env.observe(agent=player_key)['observation'].flatten()
    mask = mask=env.observe(agent=player_key)['action_mask']

    if(trainer == "random" ):
        mask_index = np.nonzero(mask)[0]
        action = np.random.choice(mask_index)
    else:
        action, _ = trainer.predict(observation = obs, mask=mask)
    
    env.step(action)
    new_obs = env.observe(agent=player_key)['observation'].flatten()

    rew_n, done_n, info_n = env.rewards, env.dones, env.infos
    player_info = info_n.get(player_key)
    done = done_n.get(player_key)
    rew_array = list(rew_n.values())
    rew = rew_array[current_agent_index]
    for agent_index, agent_reward in enumerate(rew_array):
        final_ep_ag_rewards[agent_index] += agent_reward
    
    ts += 1

    if(all(done_n.values())): # game is over
        env.reset()
        episodes += 1
        episode_step = 0

        if(episodes % 10 == 0):
            print("episode done: ", final_ep_ag_rewards, " at ts ", ts)

        continue

print(final_ep_ag_rewards)
print(episodes)