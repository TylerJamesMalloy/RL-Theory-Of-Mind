import numpy as np
from numpy.core.defchararray import index 
import torch as th 
from torch.nn import functional as F

from gym import spaces

# Todo: change this to a helper class with get functions 

class Helper():
    def __init__(self, env):
        self.env = env
        # set model type and attention size here 

    def get_input_shape(self, attention_size, model_type):
        if("texas_holdem_no_limit" in str(self.env)):
            if(model_type == "dqn"): # normal env obs shape 
                return 54
            else:
                return 108
            
        if("liars_dice" in str(self.env)):
            if(model_type == "dqn"):
                return self.env.unwrapped.obs_shape 
            else:
                return self.env.unwrapped.obs_shape + self.get_unobserved_shape()

        print("Invalid environment: ", self.env)
        assert(False)

    def get_unobserved_shape(self):
        if("texas_holdem_no_limit" in str(self.env)):
            num_players = len(self.env.agents) # currently only 2 players can be played on RLCard lib holdem 
            return 54

        if("liars_dice" in str(self.env)):
            num_opponents = len(self.env.agents) - 1
            num_sides = self.env.unwrapped.num_sides 
            num_dice = self.env.unwrapped.num_dice

            return (num_opponents * num_sides * num_dice) 

        print("Invalid environment: ", self.env)
        assert(False)
    
    def get_attention_output(self, model_type):
        if("liars_dice" in str(self.env)):
            return 24
        
        print("Invalid environment: ", self.env)
        assert(False)
    
    def get_belief(self, belief, agent):
        if("texas_holdem_no_limit" in str(self.env)):

            my_obs = self.env.observe(agent=self.env.agents[agent])['observation']
            my_cards = my_obs[:-2]
            
            print("card sum:", np.sum(my_cards))

            if(np.sum(my_cards) > 2):
                print(my_cards)
                assert(False)
            #print(self.env.observe(agent=self.env.agents[1])['observation'])
            
            return belief 
        
        if("liars_dice" in str(self.env)):
            if (belief is None):
                num_sides = self.env.unwrapped.num_sides 
                belief_probabilities = np.ones((self.get_unobserved_shape(),)) / num_sides
                return belief_probabilities
            else:
                return belief # Other hands do not depend at all on our own in this env
        
        print("Invalid environment: ", self.env)
        assert(False)

    def get_compressed_obs(self, obs, attention_idxs, attention_vals):
        if("texas_holdem_no_limit" in str(self.env)):
            # rep is already compressed, replace 1s with attention vals 

            print(obs)
            print(attention_idxs)
            print(attention_vals)

            assert(False)
            return 
        
        if("liars_dice" in str(self.env)):
            my_hand_idx = self.env.unwrapped.num_sides * self.env.unwrapped.num_dice
            my_hand = obs[0:my_hand_idx]
            others_hands = obs[self.env.unwrapped.act_shape+my_hand_idx-1:len(obs)]
            all_dice = np.concatenate((my_hand, others_hands))

            compressed_obs = np.asarray([])
            for attention_idx, attention_val in zip(attention_idxs,attention_vals):
                start = attention_idx * self.env.unwrapped.num_sides
                end = start + self.env.unwrapped.num_sides
                die_value = all_dice[start:end]
                die_location = int(attention_idx / (self.env.unwrapped.num_sides * self.env.unwrapped.num_dice))
                die_location_ohe = np.zeros(self.env.unwrapped.num_locations)
                die_location_ohe[die_location] = 1
                die_rep = np.concatenate((die_value, die_location_ohe, [attention_val]))

                compressed_obs = np.concatenate((compressed_obs, die_rep))
            action = np.asarray(obs[my_hand_idx:self.env.unwrapped.act_shape+my_hand_idx-1])
            compressed_obs = np.concatenate((compressed_obs, action))

            return compressed_obs

        print("Invalid environment: ", self.env)
        assert(False)

    def get_compressed(self, attentions, obs, attention_size, device="cuda"):
        
        print("Invalid environment: ", self.env)
        assert(False)
    


"""
    def get_input_shape(self, attention_size, model_type):
        if("mahjong" in str(self.env)):
            return attention_size * 49 # attend to some number of tiles  
        if("liars_dice" in str(self.env)):
            if(model_type == "dqn" or model_type == "belief"):
                return 612 # hard coded for 4 players, 6 dice, 6 sides for now 
            else:
                return (attention_size * (self.env.unwrapped.num_sides + self.env.unwrapped.num_locations + 1)) + self.env.unwrapped.act_shape - 1  
        

    def get_attention_shape(self):
        if("mahjong" in str(self.env)):
            return 136
        if("liars_dice" in str(self.env)):
            # attention based on attending to an entire die 
            return (self.env.num_agents * (self.env.num_agents - 1) * self.env.unwrapped.num_sides)  # each players full beliefs 

    def get_belief_shape(self):
        if("mahjong" in str(self.env)):
            return ((136 , 39), 136) # belief in -> out 
        if("liars_dice" in str(self.env)):
            belief_shape = ((self.env.unwrapped.num_agents - 1) * self.env.unwrapped.num_sides * self.env.unwrapped.num_dice)
            action_size = (self.env.unwrapped.num_bets + 2)
            return ((belief_shape, action_size), belief_shape) 
        if("texas_holdem_no_limit" in str(self.env)):
            num_players = len(self.env.agents)
            
    def get_unobserved_shape(self):
        if("mahjong" in str(self.env)):
            return (1,34,4) 
        if("liars_dice" in str(self.env)):
            return ((self.env.unwrapped.num_agents - 1) * self.env.unwrapped.num_sides * self.env.unwrapped.num_dice) 

    def get_concatonated_shape(self):
        if("mahjong" in str(self.env)):
            return (10,34,4) 
        if("liars_dice" in str(self.env)):
            hand_size = self.env.unwrapped.num_dice * self.env.unwrapped.num_sides
            num_hands = self.env.unwrapped.num_locations + self.env.num_agents 
            size = (hand_size * num_hands) + self.env.unwrapped.act_shape + 1
            size = 612 # hard code for now 
            return (size,)

    def get_compressed_obs(self, obs, attention_idxs, attention_vals):
        if("mahjong" in str(self.env)):
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
        
        if("liars_dice" in str(self.env)):
            # redo this without numpy, slows down 
            my_hand_idx = self.env.unwrapped.num_sides * self.env.unwrapped.num_dice
            my_hand = obs[0:my_hand_idx]
            others_hands = obs[self.env.unwrapped.act_shape+my_hand_idx-1:len(obs)]
            all_dice = np.concatenate((my_hand, others_hands))

            compressed_obs = np.asarray([])
            for attention_idx, attention_val in zip(attention_idxs,attention_vals):
                start = attention_idx * self.env.unwrapped.num_sides
                end = start + self.env.unwrapped.num_sides
                die_value = all_dice[start:end]
                die_location = int(attention_idx / (self.env.unwrapped.num_sides * self.env.unwrapped.num_dice))
                die_location_ohe = np.zeros(self.env.unwrapped.num_locations)
                die_location_ohe[die_location] = 1
                die_rep = np.concatenate((die_value, die_location_ohe, [attention_val]))

                compressed_obs = np.concatenate((compressed_obs, die_rep))
            action = np.asarray(obs[my_hand_idx:self.env.unwrapped.act_shape+my_hand_idx-1])
            compressed_obs = np.concatenate((compressed_obs, action))

            return compressed_obs

    def get_compressed(self, attentions, obs, attention_size, device="cuda"):
        if("mahjong" in str(self.env)):
            # most of this should be in a helper function, this is unique to mahjong 
            top_k_ind = th.topk(attentions, attention_size, dim=1).indices
            top_k_val = th.topk(attentions, attention_size, dim=1).values.unsqueeze(dim=2)

            card_val = th.remainder(top_k_ind, 34)
            card_loc = th.divide(top_k_ind, 34).to(th.int64)

            card_val_ohe = F.one_hot(card_val, num_classes=34) # one hot encoding of card value 
            card_loc_ohe = F.one_hot(card_loc, num_classes=10) # one hot encoding of card location
            
            # avoid casing-recasting by using tensors, had issues doing this previously 
            obs_numpy = obs.cpu().detach().numpy()
            card_val_numpy = card_val.cpu().detach().numpy()
            card_loc_numpy = card_loc.cpu().detach().numpy()

            card_slice = np.asarray([obs_numpy[a,card_loc_numpy[a], card_val_numpy[a]] for a in range(256)])
            card_slice = th.from_numpy(card_slice).to(device=device)
            compressed = th.cat((card_val_ohe, card_loc_ohe, card_slice, top_k_val), dim=2)
            compressed = th.flatten(compressed, start_dim=1, end_dim=2)

            return compressed
        
        if("liars_dice" in str(self.env)):
"""