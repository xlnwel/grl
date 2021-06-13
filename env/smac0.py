import numpy as np

from smac.env import StarCraft2Env
from env.smac import SMAC


def make_smac_env(config):
    config.copy()
    config['name'] = config['name'][6:]
    env = SMAC2(**config)
    return env

class SMAC2(SMAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fake_episode_steps = 0
    
    def reset(self):
        self._fake_episode_steps = 0
        obs = super().reset()
        obs['global_state'] = obs['global_state'][0]
        obs['episodic_mask'] = True
        obs.pop('life_mask')
        obs = self._get_obs(obs)
        return obs

    def step(self, action):
        obs, rewards, dones, info = super().step(action)
        obs['global_state'] = obs['global_state'][0]
        obs['episodic_mask'] = np.any(obs.pop('life_mask'))
        obs = self._get_obs(obs, action)
        reward = rewards[0]
        done = np.all(dones)
        self._fake_episode_steps += 1
        # game is over only when fake_episode_steps meets max_episode_steps
        info['game_over'] = self._fake_episode_steps >= self.max_episode_steps

        return obs, reward, done, info

    def _reset_stats(self):
        self._score = 0
        self._epslen = 0
        self._fake_episode_steps = 0

    def _get_obs(self, obs_dict, action=None):
        obs = obs_dict['obs']
        action_mask = obs_dict['action_mask']
        act_oh = np.zeros_like(action_mask).astype(np.float32)
        if obs_dict['episodic_mask'] and action is not None:
            act_oh[np.arange(self.n_agents), action] = 1
        agent_id = np.eye(self.n_agents, dtype=np.float32)
        obs = np.concatenate([obs, act_oh, agent_id], 1)
        obs_dict['obs'] = obs
        return obs_dict


# def make_smac_env(config):
#     config = config.copy()
#     config.pop('n_workers', 1)
#     config.pop('n_envs', 1)
#     config['map_name'] = config.pop('name')[6:]
#     env = SMAC2(config)
#     return env


# def info_func(agent, info):
#     if isinstance(info, list):
#         won = [i['won'] for i in info]
#     else:
#         won = info['won']
#     agent.store(win_rate=won)


# class SMAC2(gym.Env):
#     def __init__(self, config):
#         self.env = StarCraft2Env(**config)

#         env_info = self.env.get_env_info()

#         self.obs_space = (env_info['obs_shape'] + env_info['n_actions'] + env_info['n_agents'],)
#         self.obs_shape = self.obs_space
#         self.shared_state_shape = (env_info['state_shape'],)
#         self.action_space = Discrete(env_info['n_actions'])
#         self.action_shape = self.action_space.shape
#         self.action_dim = env_info['n_actions']
#         self.n_agents = env_info['n_agents']
#         self.max_episode_steps = env_info['episode_limit']
#         self.is_action_discrete = True

#         self.obs_dtype = np.float32
#         self.shared_state_dtype = np.float32
#         self.action_dtype = np.int32

#         self._reset_stats()
#         self._has_done = False
#         self._force_restarts = self.env.force_restarts
    
#     def random_action(self):
#         actions = []
#         for avail_actions in self.env.get_avail_actions():
#             choices = [i for i, v in enumerate(avail_actions) if v]
#             actions.append(random.choice(choices))
            
#         return np.stack(actions)

#     def reset(self):
#         self._reset_stats()
#         self._has_done = False

#         self.env.reset()
#         obs = self._get_obs()
#         if not hasattr(self, '_empty_obs'):
#             self._empty_obs = {k: np.zeros_like(v) for k, v in obs.items()}

#         return obs

#     def step(self, action):
#         if self._has_done:
#             obs = self._get_obs()
#             obs = self._empty_obs
#             obs['episodic_mask'] = 0.
#             reward = 0
#             done = True
#             info = self.info
#         else:
#             reward, done, info = self.env.step(action)
#             obs = self._get_obs(action)
#             self._has_done = done
#             self._score += reward
#             self._epslen += 1
#             if done:
#                 info['score'] = self._score
#                 info['epslen'] = self._epslen
#                 info['won'] = info['battle_won']
#             self.info = info

#         self._fake_episode_steps += 1
#         # game is over only when fake_episode_steps meets max_episode_steps
#         info['game_over'] = self._fake_episode_steps >= self.max_episode_steps
#         if self._force_restarts != self.env.force_restarts:
#             print('Forcing restart due to an exception')
#             self._force_restarts = self.env.force_restarts

#         return obs, reward, done, info

#     def _reset_stats(self):
#         self._score = 0
#         self._epslen = 0
#         self._fake_episode_steps = 0

#     def _get_obs(self, action=None):
#         obs = np.array(self.env.get_obs(), np.float32)
#         action_mask = np.array(self.env.get_avail_actions(), np.bool)
#         act_oh = np.zeros_like(action_mask).astype(np.float32)
#         if action is not None:
#             act_oh[np.arange(self.n_agents), action] = 1
#         agent_id = np.eye(self.n_agents, dtype=np.float32)
#         obs = np.concatenate([obs, act_oh, agent_id], 1)
#         obs = dict(
#             obs=obs,
#             global_state=self.env.get_state(),
#             action_mask=action_mask,
#             episodic_mask=1.,
#         )
#         return obs


# if __name__ == '__main__':
#     config = dict(
#         name='smac2_3m',
#         obs_last_action=False
#     )
#     env = make_smac_env(config)
#     env.reset()
#     for k in range(100):
#         o, r, d, i = env.step(env.random_action())
#         if i['game_over']:
#             print('reset', env._fake_episode_steps)
#             print(i['score'], i['epslen'])
#             o = env.reset()
#         print(k, d, o['obs'][0].shape)

if __name__ == '__main__':
    config = dict(
        name='smac2_3m',
        obs_last_action=False
    )
    env = make_smac_env(config)
    env.reset()
    for k in range(100):
        o, r, d, i = env.step(env.random_action())
        if i['game_over']:
            print('reset', env._fake_episode_steps)
            print(i['score'], i['epslen'])
            o = env.reset()
        print(k, d, o['obs'][0].shape)