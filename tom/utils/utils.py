import numpy as np 
import torch as th 
from torch.nn import functional as F

from gym import spaces

# Todo: change this to a helper class with get functions 

def get_input_shape(env, attention_size):
    if("mahjong" in str(env)):
        return attention_size * 49
    
def get_attention_shape(env):
    if("mahjong" in str(env)):
        return 136

def get_belief_shape(env):
    if("mahjong" in str(env)):
        return ((136 , 39), 136) # belief in -> out 

def get_unobserved_shape(env):
    if("mahjong" in str(env)):
        return (1,34,4)  

def get_concatonated_shape(env):
    if("mahjong" in str(env)):
        return (10,34,4) 

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

#only for mahjong
def get_compressed(attentions, obs, attention_size, device="cuda"):
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

    card_slice = np.asarray([obs_numpy[a,card_loc_numpy[a], card_val_numpy[a]] for a in range(16)])
    card_slice = th.from_numpy(card_slice).to(device=device)
    compressed = th.cat((card_val_ohe, card_loc_ohe, card_slice, top_k_val), dim=2)
    compressed = th.flatten(compressed, start_dim=1, end_dim=2)

    return compressed

