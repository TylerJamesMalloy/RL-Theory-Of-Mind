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
        handle_timeout_termination: bool = True,
    ):
        self.n_envs = 1
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.action_dim = len(action_space.shape)
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.masks = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env):
        batch_inds = random.choices(np.linspace(0,self.buffer_size-1, num=self.buffer_size), k=batch_size)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            self._normalize_obs(self.next_observations[batch_inds, 0, :], env),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
        )
        return tuple(map(self.to_torch, data))