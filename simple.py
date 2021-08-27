from pettingzoo.tom import liars_dice_v0 
from pettingzoo.classic import mahjong_v4
from tom.matom import MATOM
import numpy as np
import torch as th 
import argparse
import time 
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser(description='Options for Multi-Agent Theory of Mind Model.')
parser.add_argument('--model_type', default='full', type=str, help='full, attention, belief, dqn')
parser.add_argument('--folder', default='./', type=str, help='Location to save all agent models')
parser.add_argument('--environment', default='liars_dice', type=str, help='Environment to train agents in')
parser.add_argument('--timesteps', default=100000, type=int, help='Timesteps for training')
parser.add_argument('--compare', default=False, type=bool, help='Timesteps for training')
parser.add_argument('--first_folder', default='./', type=str, help='first load model location')
parser.add_argument('--second_folder', default='./', type=str, help='second load model location')

args = parser.parse_args()

env = liars_dice_v0.env(num_players=4)
env.reset()

# python simple.py --compare True --first_folder ./model/liarsDice/full/t1000 --second_folder ./model/liarsDice/full/t1000 --timesteps 1000


# python simple.py --model_type full --folder ./model/liarsDice/full/t1000 --timesteps 1000
# python simple.py --model_type attention --folder ./model/liarsDice/attention/t1000 --timesteps 1000
# python simple.py --model_type belief --folder ./model/liarsDice/belief/t1000 --timesteps 1000
# python simple.py --model_type dqn --folder ./model/liarsDice/dqn/t1000 --timesteps 1000

if(args.compare):
    agent1 = args.first_folder + "/player_player_0/"
    agent2 = args.second_folder + "/player_player_1/"
    agent3 = args.second_folder + "/player_player_2/"
    agent4 = args.second_folder + "/player_player_3/"
    load_paths = [agent1, agent2, agent3, agent4]

    model = MATOM(env, model_type=None, load_models=True, load_paths=load_paths, load_types=["full", "random", "random", "random"])

    results = model.learn(args.timesteps, train=False)
    results = np.asarray(results)
    print(results.shape)
    print(np.sum(results, axis=0))

    assert(False)
    
else:
    model = MATOM(env, model_type=args.model_type)
    start_time = time.time()

    model.learn(args.timesteps)
    model.save(args.folder)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("done")

# train 1K
# test on random agent 