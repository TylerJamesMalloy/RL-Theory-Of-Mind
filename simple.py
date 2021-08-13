from pettingzoo.tom import liars_dice_v0 
from pettingzoo.classic import mahjong_v3
from tom.matom import MATOM
import numpy as np
import torch as th 
import time 



env = mahjong_v3.env()
env.reset()


model = MATOM(env)

start_time = time.time()

model.learn(100000)
model.save("./model/t100000")

print("--- %s seconds ---" % (time.time() - start_time))

print("done")