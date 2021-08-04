import numpy as np 


def get_input_shape(env, attention_size):
    if("mahjong" in str(env)):
        return attention_size * 40
    
def get_attention_shape(env):
    if("mahjong" in str(env)):
        return 136

def get_belief_shape(env):
    if("mahjong" in str(env)):
        return ((136 , 37), 136) # belief in -> out 

def get_compressed_obs(env, obs, attention_idxs):
    if("mahjong" in str(env)):
        print(obs)
        print(attention_idxs)

        np.ones(136)

