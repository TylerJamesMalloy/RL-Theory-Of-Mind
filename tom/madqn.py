from os import stat
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch import nn
import gym
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update, obs_as_tensor, safe_mean, should_collect_more_steps
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

class MADQN():
    def __init__(self, env):
        self.env = env 
        agents = []

        for agent_key in env.agents:
            setattr(env, "action_space", env.action_spaces[agent_key])
            setattr(env, "observation_space", env.observation_spaces[agent_key]['observation'])

            if(hasattr(env.unwrapped, 'reward_range')):
                setattr(env, "reward_range", env.unwrapped.reward_range)
            else:
                setattr(env, "reward_range", (0,1))

            agents.append(DQN("MlpPolicy", env))
            
        self.agents = agents

    def learn(self, timesteps):
        env = self.env

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(len(self.agents))]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        episode_step = 0
        env.reset()
        ts = 0

        total_timesteps, callback = self.agents[0]._setup_learn(
            total_timesteps=timesteps,
            eval_env=env,
            callback=None,
        )

        print(callback)

        while True:
            current_agent_index = env.agents.index(env.agent_selection)
            episode_step += 1
            trainer = self.agents[current_agent_index]
            player_key = env.agents[current_agent_index]

            if(ts > timesteps):
                return final_ep_ag_rewards

            obs = env.observe(agent=player_key)['observation'].flatten()
            mask = mask=env.observe(agent=player_key)['action_mask']
            action, _ = trainer.predict(observation = obs, mask=mask)
            
            env.step(action)
            new_obs = env.observe(agent=player_key)['observation'].flatten()

            rew_n, done_n, info_n = env.rewards, env.dones, env.infos
            player_info = info_n.get(player_key)
            done = done_n.get(player_key)
            rew_array = list(rew_n.values())
            rew = rew_array[current_agent_index]
            for agent_index, agent_reward in enumerate(rew_array):
                agent_rewards[agent_index][0] += agent_reward
            ts += 1

            # Store data in replay buffer
            trainer.replay_buffer.add(obs,new_obs,action,rew,done,[player_info])
            trainer._on_step()
            #trainer.train(batch_size=trainer.batch_size, gradient_steps=trainer.gradient_steps)

            from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: env])

            rollout = trainer.collect_rollouts(
                vec_env,
                train_freq=trainer.train_freq,
                action_noise=trainer.action_noise,
                callback=callback,
                learning_starts=trainer.learning_starts,
                replay_buffer=trainer.replay_buffer,
                log_interval=100,
            )

            if rollout.continue_training is False:
                break

            if trainer.num_timesteps > 0 and trainer.num_timesteps > trainer.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = trainer.gradient_steps if trainer.gradient_steps > 0 else rollout.episode_timesteps
                trainer.train(batch_size=trainer.batch_size, gradient_steps=gradient_steps)
            

            if(all(done_n.values())): # game is over
                final_ep_rewards.append(episode_rewards) 
                final_ep_ag_rewards.append(agent_rewards)

                episode_rewards = [0.0]  # sum of rewards for all agents
                agent_rewards = [[0.0] for _ in range(len(self.agents))]  # individual agent reward

                env.reset()
                continue

    def save(self, folder):
        for agent_id, agent in enumerate(self.agents):
            agent.save(folder + "/" + str(agent_id))

class DQN(OffPolicyAlgorithm):
    """
    See stable baselines for info
    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 5000,
        batch_size: Optional[int] = 1024,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 10,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(DQN, self).__init__(
            policy,
            env,
            DQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(DQN, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to allow for action masking 

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        if not deterministic and np.random.rand() < self.exploration_rate:
            # need to only sample from valid actions: 
            
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                mask_indicies = np.array([np.nonzero(sub_mask) for sub_mask in mask])
                #action = np.array([self.action_space.sample() for _ in range(n_batch)])
                action = np.array([np.random.choice(mask_index) for mask_index in mask_indicies])
            else:
                #action = np.array(self.action_space.sample())
                mask_index = np.nonzero(mask)[0]
                action = np.random.choice(mask_index)
        else:
            #action, state = self.policy.predict(observation, state, mask, deterministic)
            q_values = self.q_net.forward(th.unsqueeze(obs_as_tensor(observation, self.device), 0))
            policy = self.masked_softmax(q_values, th.tensor(mask))

            np.set_printoptions(threshold=np.inf)
            action = np.random.choice(a=np.linspace(0,len(policy)-1, num=len(policy), dtype=int), size=1, p=policy)[0]

        return action, None
    
    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        exps = th.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps/masked_sums).detach().cpu().numpy()[0]

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(DQN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _excluded_save_params(self) -> List[str]:
        return super(DQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

