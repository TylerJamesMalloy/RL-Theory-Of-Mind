from pettingzoo.tom import liars_dice_v0 
from pettingzoo.classic import mahjong_v3
from tom.matom import MATOM
import numpy as np
import time 


env = mahjong_v3.env()
env.reset()


model = MATOM(env)

model.learn(100000)