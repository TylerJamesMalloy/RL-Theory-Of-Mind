import argparse, multiprocessing
from pettingzoo.tom import liars_dice_v0 
from pettingzoo.classic import backgammon_v3
from tom.madqn import MADQN
import numpy as np
import time 

#env = liars_dice_v0.env(num_players=2, num_dice=5, num_sides=6, common_hand=False, opponents_hand_visible=False)
env = backgammon_v3.env()
env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # arguments for experiment setting
    parser.add_argument('--training_ts', type=int, default=1000000, help='Timesteps to train')
    parser.add_argument('--model_folder', type=str, default='Model/1M', help='Timesteps to train')
    
    args = parser.parse_args()

    start_time = time.time()

    model = MADQN(env)
    rewards = model.learn(args.training_ts)

    print("--- %s seconds ---" % (time.time() - start_time))

    model.save(args.model_folder)


# get a player and play them against an opponent that makes random selections