from typing import Tuple
import numpy as np

### Date : 2025-10-28
### Reference :  ElegantRL @Github

class ReplayBuffer:
    def __init__(self, cfg):
        self.buffer_size = cfg.buffer_size
        self.pointer = 0
        self.current_size = 0
        self.is_full = self.current_size == self.buffer_size
        self.is_discrete = cfg.is_discrete
        self.obs_dim = cfg.observation_dim
        self.num_envs = cfg.num_envs
        self.action_dim = 1 if cfg.is_discrete else cfg.action_dim


        # Create in CPU, np
        # action, reward, mask, action_prob/action_noise
        # action_dim , 1, 1, action_dim
        self.observations = np.empty((self.buffer_size, self.num_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.empty((self.buffer_size, self.num_envs, ), dtype=np.float32) if cfg.is_discrete \
            else np.empty((self.buffer_size, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.buffer_size, self.num_envs), dtype=np.float32)
        self.unmasks = np.empty((self.buffer_size, self.num_envs), dtype=np.float32) # Not Truncated
        self.undone = np.empty((self.buffer_size, self.num_envs), dtype=np.float32) # Not Terminated

        if cfg.is_discrete:
            self.action_log_probs = np.empty((self.buffer_size, self.num_envs, self.action_dim), dtype=np.float32)
        else:
            self.action_noises  = np.empty((self.buffer_size, self.num_envs, self.action_dim), dtype=np.float32)


    def update_buffer_horizon(self, items: Tuple):
        observations, actions, rewards, undone, unmasks = items

        assert rewards.shape[1:] == (self.num_envs,)
        assert actions.shape[1:] == (self.num_envs, ) if self.is_discrete else (self.num_envs, self.action_dim)
        assert observations.shape[1:] == (self.num_envs, self.obs_dim)

        data_size = rewards.shape[0]

        expect_len = data_size + self.pointer

        if expect_len > self.buffer_size:
            self.is_full = True

            replace_len = expect_len - self.buffer_size
            fill_len = self.buffer_size - self.pointer

            # replace the first expect_len - max_len
            self.observations[self.pointer:self.buffer_size], self.observations[:replace_len] = observations[:fill_len], observations[-replace_len:]
            self.actions[self.pointer:self.buffer_size], self.actions[:replace_len] = actions[:fill_len], actions[-replace_len:]
            self.rewards[self.pointer:self.buffer_size], self.rewards[:replace_len] = rewards[:fill_len], rewards[-replace_len:]
            self.undone[self.pointer:self.buffer_size], self.undone[:replace_len] = undone[:fill_len], undone[-replace_len:]
            self.unmasks[self.pointer:self.buffer_size], self.unmasks[:replace_len] = unmasks[:fill_len], unmasks[-replace_len:]

            expect_len = replace_len

            self.current_size = self.buffer_size

        else:
            self.is_full = expect_len == self.buffer_size

            # fill
            self.observations[self.pointer:expect_len] = observations
            self.actions[self.pointer:expect_len] = actions
            self.rewards[self.pointer:expect_len] = rewards
            self.undone[self.pointer:expect_len] = undone
            self.unmasks[self.pointer:expect_len] = unmasks

            self.current_size = expect_len

        self.pointer = expect_len

    def update_buffer_step(self, items: Tuple):
        observations, actions, rewards, undone, unmasks = items

        assert rewards.shape ==  (self.num_envs,)
        assert actions.shape == (self.num_envs, ) if self.is_discrete else (self.num_envs, self.action_dim)
        assert observations.shape == (self.num_envs, self.obs_dim)

        self.observations[self.pointer] = observations
        self.actions[self.pointer] = actions
        self.rewards[self.pointer] = rewards
        self.undone[self.pointer] = undone
        self.unmasks[self.pointer] = unmasks
        self.pointer = (self.pointer + 1) % self.buffer_size


        if not self.is_full:
            self.current_size += 1


        self.is_full = self.current_size == self.buffer_size



    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        assert self.current_size > 1
        sample_len = self.current_size - 1
        idx = np.random.randint(sample_len * self.num_envs, size=  (batch_size,))

        ids0 = np.fmod(idx, sample_len)
        ids1 = np.floor_divide(idx, sample_len)
        # (time_step, num_envs*num_agents, obs_dim)
        return self.observations[ids0, ids1], self.actions[ids0, ids1], self.rewards[ids0, ids1], self.undone[ids0, ids1], self.unmasks[ids0, ids1], self.observations[ids0+1, ids1]


    def __len__(self):
        return self.current_size




