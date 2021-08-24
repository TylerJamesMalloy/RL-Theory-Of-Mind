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

args = parser.parse_args()

env = liars_dice_v0.env(num_players=4)
env.reset()

model = MATOM(env, model_type=args.model_type)

start_time = time.time()

model.learn(args.timesteps)
model.save(args.folder)

# python simple.py --model_type dqn --folder ./model/liarsDice/dqn/t10000 --timesteps 10000

print("--- %s seconds ---" % (time.time() - start_time))

print("done")

# train 1K
# test on random agent 