import numpy as np 

# Todo: change this to a helper class with get functions 

def get_input_shape(env, attention_size):
    if("mahjong" in str(env)):
        return attention_size * 48
    
def get_attention_shape(env):
    if("mahjong" in str(env)):
        return 136

def get_belief_shape(env):
    if("mahjong" in str(env)):
        return ((136 , 37), 136) # belief in -> out 

def get_unobserved_shape(env):
    if("mahjong" in str(env)):
        return (1,34,4)  

def get_concatonated_shape(env):
    if("mahjong" in str(env)):
        return (10,34,4) 

def get_compressed_obs(env, obs, attention_idxs):
    if("mahjong" in str(env)):
        compressed_obs = []

        # TODO: perform logic on beliefs? 

        for tile in attention_idxs:
            tile_index = tile % 34
            hand_index = int(tile / 34)
            tile_rep = np.zeros(48)
            tile_rep[tile_index] = 1                    # set tile type identifier to 1 
            tile_rep[hand_index + 34] = 1               # set the hand type identifier to 1 
            tile_rep[-4:] = obs[hand_index][tile_index] # set last 4 values to the values of the tile presence

            compressed_obs.append(tile_rep)

        compressed_obs = np.asarray(compressed_obs).flatten()
        return compressed_obs

