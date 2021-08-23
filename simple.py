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

args = parser.parse_args()

env = liars_dice_v0.env(num_players=4)
env.reset()

model = MATOM(env, model_type="full")

start_time = time.time()

model.learn(5000)
model.save("./model/t5000")

print("--- %s seconds ---" % (time.time() - start_time))

print("done")

# train 1K

# test on random agent 