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
parser.add_argument('--timesteps', default=25000, type=int, help='Timesteps for training')
parser.add_argument('--compare', default=False, type=bool, help='Timesteps for training')
parser.add_argument('--first_folder', default='./', type=str, help='first load model location')
parser.add_argument('--second_folder', default='./', type=str, help='second load model location')
parser.add_argument('--num_players', default=2, type=int, help='number of models to train or load, must match folder when loading')

args = parser.parse_args()

env = liars_dice_v0.env(num_players=args.num_players)
env.reset()

# python simple.py --folder ./models/liarsDice4/full/t1000/ --model_type full --timesteps 1000 --num_players 4 
# python simple.py --folder ./models/liarsDice4/belief/t1000/ --model_type belief --timesteps 1000 --num_players 4 
# python simple.py --folder ./models/liarsDice4/attention/t1000/ --model_type attention --timesteps 1000 --num_players 4 
# python simple.py --folder ./models/liarsDice4/dqn/t1000/ --model_type dqn --timesteps 1000 --num_players 4 

# python simple.py --compare True  --first_folder ./models/liarsDice4/full/t1000 --second_folder ./models/liarsDice4/dqn/t1000 --timesteps 1000 --num_players 4

# Full v. DQN: 
# Full v. Belief: 
# Full v. Attent: 
# Full v. Random 

if(args.compare):
    agent1 = args.first_folder  + "/player_player_0/"
    agent2 = args.second_folder + "/player_player_1/"
    agent3 = args.second_folder + "/player_player_2/"
    agent4 = args.second_folder + "/player_player_3/"
    load_paths = [agent1, agent2, agent3, agent4]

    model = MATOM(env, model_type=None, load_models=True, load_paths=load_paths, load_types=["full", "dqn", "dqn", "dqn"])

    results = model.learn(args.timesteps, train=False)
    results = np.asarray(results)

    player1 = results[0].flatten()
    player2 = results[1].flatten()
    player3 = results[2].flatten()
    player4 = results[3].flatten()

    games = len(player1)

    player1_wins = np.count_nonzero(player1 == 1)
    player1_loss = np.count_nonzero(player1 == -1)

    player2_wins = np.count_nonzero(player2 == 1)
    player2_loss = np.count_nonzero(player2 == -1)

    player3_wins = np.count_nonzero(player3 == 1)
    player3_loss = np.count_nonzero(player3 == -1)

    player4_wins = np.count_nonzero(player4 == 1)
    player4_loss = np.count_nonzero(player4 == -1)
    
    print("Player 1 won: ",  player1_wins, " times out of ", games, " games ")
    print("Player 1 lost: ",  player1_wins, " times out of ", games, " games ")

    print("Player 2 won: ",  player2_wins, " times out of ", games, " games ")
    print("Player 2 lost: ",  player2_wins, " times out of ", games, " games ")

    print("Player 3 won: ",  player3_wins, " times out of ", games, " games ")
    print("Player 3 lost: ",  player3_wins, " times out of ", games, " games ")

    print("Player 4 won: ",  player4_wins, " times out of ", games, " games ")
    print("Player 4 losy: ",  player4_wins, " times out of ", games, " games ")

    results = np.sum(results, axis=0)
    

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