import numpy as np
import torch as th
import random 
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F
from typing import Any, Dict, Generator, List, Optional, Union

class ReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        representation_space: spaces.Space,
        belief_shape: tuple,
        handle_timeout_termination: bool = True,
    ):
        self.n_envs = 1
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.rep_shape = representation_space.shape 
        self.action_dim = len(action_space.shape)
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.representations = np.zeros((self.buffer_size, self.n_envs) + self.rep_shape, dtype=representation_space.dtype)
        self.next_representations = np.zeros((self.buffer_size, self.n_envs) + self.rep_shape, dtype=representation_space.dtype)
        self.masks = np.zeros((self.buffer_size, self.n_envs, action_space.n), dtype=np.int32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=action_space.dtype)
        self.prev_actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=action_space.dtype)
        self.prev_beliefs = np.zeros((self.buffer_size, belief_shape[0], belief_shape[1]), dtype=action_space.dtype) 
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        rep: np.ndarray,
        next_rep: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        prev_belief: np.ndarray,
        mask:np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.representations[self.pos] = np.array(rep).copy()
        self.next_representations[self.pos] = np.array(next_rep).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.prev_actions[self.pos] = np.array(prev_action).copy()
        self.prev_beliefs[self.pos] = np.array(prev_belief).copy()
        self.masks[self.pos] = np.array(mask).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def size(self):
        return self.pos 

    def sample(self, batch_size: int, env):
        #batch_inds = random.choices(np.linspace(0,self.buffer_size-1, num=self.buffer_size), k=batch_size)
        batch_inds = random.choices(np.linspace(0,self.pos-1, num=self.pos, dtype=np.int32), k=batch_size)

        data = (
            self.normalize_obs(self.observations[batch_inds, 0, :], env),
            self.normalize_obs(self.next_observations[batch_inds, 0, :], env),
            self.representations[batch_inds, 0, :],
            self.next_representations[batch_inds, 0, :],
            self.actions[batch_inds, 0, :],
            self.prev_actions[batch_inds, 0, :],
            self.prev_beliefs[batch_inds, 0, :],
            self.masks[batch_inds, 0, :],
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self.normalize_reward(self.rewards[batch_inds], env),
        )
        return tuple(map(self.to_torch, data))
    
    def normalize_obs(self, obs, env):
        return obs 
    
    def normalize_reward(self, reward, env):
        return reward 
    
    def to_torch(self, input):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        return th.from_numpy(input).to(device)