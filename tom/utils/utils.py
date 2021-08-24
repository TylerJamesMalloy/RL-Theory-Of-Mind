import numpy as np
from numpy.core.defchararray import index 
import torch as th 
from torch.nn import functional as F

from gym import spaces

# Todo: change this to a helper class with get functions 

def get_input_shape(env, attention_size, model_type):
    if("mahjong" in str(env)):
        return attention_size * 49 # attend to some number of tiles  
    if("liars_dice" in str(env)):
        if(model_type == "dqn" or model_type == "attnetion"):
            return 612 # hard coded for 4 players, 6 dice, 6 sides for now 
        else:
            return (attention_size * (env.unwrapped.num_sides + env.unwrapped.num_locations + 1)) + env.unwrapped.act_shape - 1  
    
def get_attention_shape(env):
    if("mahjong" in str(env)):
        return 136
    if("liars_dice" in str(env)):
        # attention based on attending to an entire die 
        return (env.num_agents * (env.num_agents - 1) * env.unwrapped.num_sides)  # each players full beliefs 

def get_belief_shape(env):
    if("mahjong" in str(env)):
        return ((136 , 39), 136) # belief in -> out 
    if("liars_dice" in str(env)):
        belief_shape = ((env.unwrapped.num_agents - 1) * env.unwrapped.num_sides * env.unwrapped.num_dice)
        action_size = (env.unwrapped.num_bets + 2)
        return ((belief_shape, action_size), belief_shape) 

def get_unobserved_shape(env):
    if("mahjong" in str(env)):
        return (1,34,4) 
    if("liars_dice" in str(env)):
        return ((env.unwrapped.num_agents - 1) * env.unwrapped.num_sides * env.unwrapped.num_dice) 

def get_concatonated_shape(env):
    if("mahjong" in str(env)):
        return (10,34,4) 
    if("liars_dice" in str(env)):
        hand_size = env.unwrapped.num_dice * env.unwrapped.num_sides
        num_hands = env.unwrapped.num_locations + env.num_agents 
        size = (hand_size * num_hands) + env.unwrapped.act_shape + 1
        size = 612 # hard code for now 
        return (size,)

def get_compressed_obs(env, obs, attention_idxs, attention_vals):
    if("mahjong" in str(env)):
        compressed_obs = []

        # TODO: perform logic on beliefs? 
        for index, tile in enumerate(attention_idxs):
            tile_index = tile % 34
            hand_index = int(tile / 34)
            tile_rep = np.zeros(49)
            tile_rep[tile_index] = 1                        # set tile type identifier to 1 
            tile_rep[hand_index + 34] = 1                   # set the hand type identifier to 1 
            tile_rep[-5:-1] = obs[hand_index][tile_index]   # set last 5-2 values to the values of the tile presence
            tile_rep[-1:] = attention_vals[index]           # set last 1 value to the attention value 

            compressed_obs.append(tile_rep)

        compressed_obs = np.asarray(compressed_obs).flatten()
        return compressed_obs
    
    if("liars_dice" in str(env)):
        my_hand_idx = env.unwrapped.num_sides * env.unwrapped.num_dice
        my_hand = obs[0:my_hand_idx]
        others_hands = obs[env.unwrapped.act_shape+my_hand_idx-1:len(obs)]
        all_dice = np.concatenate((my_hand, others_hands))

        compressed_obs = np.asarray([])
        for attention_idx, attention_val in zip(attention_idxs,attention_vals):
            start = attention_idx * env.unwrapped.num_sides
            end = start + env.unwrapped.num_sides
            die_value = all_dice[start:end]
            die_location = int(attention_idx / (env.unwrapped.num_sides * env.unwrapped.num_dice))
            die_location_ohe = np.zeros(env.unwrapped.num_locations)
            die_location_ohe[die_location] = 1
            die_rep = np.concatenate((die_value, die_location_ohe, [attention_val]))

            compressed_obs = np.concatenate((compressed_obs, die_rep))
        action = np.asarray(obs[my_hand_idx:env.unwrapped.act_shape+my_hand_idx-1])
        compressed_obs = np.concatenate((compressed_obs, action))

        return compressed_obs


#only for mahjong
def get_compressed(env, attentions, obs, attention_size, device="cuda"):
    if("mahjong" in str(env)):
        # most of this should be in a helper function, this is unique to mahjong 
        top_k_ind = th.topk(attentions, attention_size, dim=1).indices
        top_k_val = th.topk(attentions, attention_size, dim=1).values.unsqueeze(dim=2)

        card_val = th.remainder(top_k_ind, 34)
        card_loc = th.divide(top_k_ind, 34).to(th.int64)

        card_val_ohe = F.one_hot(card_val, num_classes=34) # one hot encoding of card value 
        card_loc_ohe = F.one_hot(card_loc, num_classes=10) # one hot encoding of card location
        
        # TODO: avoid casing-recasting by using tensors, had issues doing this previously 
        obs_numpy = obs.cpu().detach().numpy()
        card_val_numpy = card_val.cpu().detach().numpy()
        card_loc_numpy = card_loc.cpu().detach().numpy()

        card_slice = np.asarray([obs_numpy[a,card_loc_numpy[a], card_val_numpy[a]] for a in range(256)])
        card_slice = th.from_numpy(card_slice).to(device=device)
        compressed = th.cat((card_val_ohe, card_loc_ohe, card_slice, top_k_val), dim=2)
        compressed = th.flatten(compressed, start_dim=1, end_dim=2)

        return compressed
    
    if("liars_dice" in str(env)):
        top_k_ind = th.topk(attentions, attention_size, dim=1).indices
        top_k_val = th.topk(attentions, attention_size, dim=1).values.unsqueeze(dim=2)

        my_hand_idx = env.unwrapped.num_sides * env.unwrapped.num_dice
        my_hand = obs[:, 0:my_hand_idx]
        others_hands = obs[:, env.unwrapped.act_shape+my_hand_idx-1:obs.shape[1]]
        all_dice = th.cat((my_hand, others_hands), dim=1)

        compressed_obs = []
        for batch in range(top_k_ind.shape[0]):
            attention_idxs = top_k_ind[batch]
            attention_vals = top_k_val[batch]
            batch_dice = all_dice[batch]
            batch_obs = th.from_numpy(np.asarray([])).to(device=device)
            for attention_idx, attention_val in zip(attention_idxs,attention_vals):
                start = attention_idx * env.unwrapped.num_sides
                end = start + env.unwrapped.num_sides
                die_value = batch_dice[start:end]
                die_location = th.tensor(int(attention_idx / (env.unwrapped.num_sides * env.unwrapped.num_dice)))
                die_location_ohe = F.one_hot(die_location, num_classes=env.unwrapped.num_locations).to(device=device)
                die_rep = th.cat((die_value, die_location_ohe, attention_val))

                batch_obs = th.cat((batch_obs, die_rep))
            
            action = obs[batch, my_hand_idx:env.unwrapped.act_shape+my_hand_idx-1]
            batch_obs = th.cat((batch_obs, action))
            compressed_obs.append(batch_obs)
        
        compressed_obs = th.stack(compressed_obs)
        return compressed_obs
    


