from pettingzoo.tom import liars_dice_v0, commune_v0
from pettingzoo.classic import mahjong_v4, texas_holdem_no_limit_v5
from tom.matom import MATOM
import numpy as np
import torch as th 
import argparse
import time 
import os

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser(description='Options for Multi-Agent Theory of Mind Model.')
parser.add_argument('--model_type', default='full', type=str, help='full, attention, belief, dqn')
parser.add_argument('--folder', default='./', type=str, help='Location to save all agent models')
parser.add_argument('--environment', default='liars_dice', type=str, help='Environment to train agents in')
parser.add_argument('--timesteps', default=1000, type=int, help='Timesteps for training')
parser.add_argument('--compare', default=False, type=bool, help='Timesteps for training')
parser.add_argument('--first_folder', default='./', type=str, help='first load model location')
parser.add_argument('--second_folder', default='./', type=str, help='second load model location')
parser.add_argument('--num_players', default=2, type=int, help='number of models to train or load, must match folder when loading')

args = parser.parse_args()

env = liars_dice_v0.env(num_players=4)

if(args.compare):
    agent1 = args.second_folder  + "/player_player_0/"
    agent2 = args.first_folder + "/player_player_1/"
    agent3 = args.second_folder + "/player_player_2/"
    agent4 = args.second_folder + "/player_player_3/"
    load_paths = [agent1, agent2, agent3, agent4]

    model = MATOM(env, model_type=None, load_models=True, load_paths=load_paths, load_types=["full", "up", "up", "up"])

    results = model.learn(args.timesteps, train=False)
    results = np.asarray(results)

    print(results.shape)

    player1 = results[:,0].flatten()
    player2 = results[:,1].flatten()
    player3 = results[:,2].flatten()
    player4 = results[:,3].flatten()

    games = len(player1)

    player1_wins = np.count_nonzero(player1 == 1)
    player2_wins = np.count_nonzero(player2 == 1)
    player3_wins = np.count_nonzero(player3 == 1)
    player4_wins = np.count_nonzero(player4 == 1)

    print("Player 1 won: ",  player1_wins, " times out of ", games, " games ", player1_wins/games, " percent")
    print("Player 2 won: ",  player2_wins, " times out of ", games, " games ", player2_wins/games, " percent")
    print("Player 3 won: ",  player3_wins, " times out of ", games, " games ", player3_wins/games, " percent")
    print("Player 4 won: ",  player4_wins, " times out of ", games, " games ", player4_wins/games, " percent")

    results = np.sum(results, axis=0)
    

    assert(False)
    
else:
    model = MATOM(env, model_type=args.model_type)
    start_time = time.time()

    model.learn(args.timesteps)
    model.save(args.folder)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("done")

