import metaworld
import random

from gym import spaces as gym_spaces
import numpy as np

try:
    import pybullet_envs
    import time
    pybullet_found = True
except ImportError:
    pybullet_found = False

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

def make_env(env_name, 
             env_cls, 
             train_tasks, 
             horizon=500, 
             gamma=0.99, 
             normalize_reward=False, 
             sample_task_per_episode=False):
    
    _env = env_cls()
    task_list = [task for task in train_tasks if task.env_name == env_name]
    _env.set_task(random.choice(task_list))

    action_space = MetaWorldEnv._convert_gym_space(_env.action_space)
    observation_space = MetaWorldEnv._convert_gym_space(_env.observation_space)

    def _init():
        env = MetaWorldTaskEnv( _env, env_name=env_name,
                                observation_space=observation_space, action_space=action_space,
                                horizon=horizon, gamma = gamma, task_list=task_list, 
                                normalize_reward=normalize_reward, sample_task_per_episode=sample_task_per_episode)
        return env

    return _init

class MetaWorldEnv(Environment):
    """
    Interface for Metaworld environments. It makes it possible to use every
    metaworld environment just providing the env id and experiment type.
    """

    def __init__(self, benchmark_name,
                       env_name = None,
                       horizon=500, 
                       gamma=0.99, 
                       wrappers=None, 
                       wrappers_args=None, 
                       normalize_reward = False, 
                       sample_task_per_episode = False, 
                       render_mode = "rgb_array",
                       **env_args):
        # MDP creation
        self._first = True

        self._training_envs = []
        self._env_names = []

        # experiment type [MT1, ML1, MT5, MT10, ML5, ML10]
        if "MT1" == benchmark_name or "ML1" == benchmark_name:
            self.exp = getattr(metaworld, benchmark_name)(env_name)
        else:
            self.exp = getattr(metaworld, benchmark_name)()

        for env_name, env_cls in self.exp.train_classes.items():
            env = env_cls() #render_mode=render_mode

            task_list = [task for task in self.exp.train_tasks if task.env_name == env_name]
            env.set_task(random.choice(task_list))
            
            # wrappers
            if wrappers is not None:
                if wrappers_args is None:
                    wrappers_args = [dict()] * len(wrappers)
                for wrapper, args in zip(wrappers, wrappers_args):
                    if isinstance(wrapper, tuple):
                        env = wrapper[0](env, *args, **wrapper[1])
                    else:
                        env = wrapper(env, *args, **env_args)
        
            action_space = self._convert_gym_space(env.action_space)
            observation_space = self._convert_gym_space(env.observation_space)

            
            self._env_names.append(env_name)
            self._training_envs.append(MetaWorldTaskEnv(env, env_name=env_name,
                                                         observation_space=observation_space, action_space=action_space,
                                                         horizon=horizon, gamma = gamma, task_list=task_list, 
                                                         normalize_reward=normalize_reward, sample_task_per_episode=sample_task_per_episode))

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._normalize_reward = normalize_reward
        self._sample_task_per_episode = sample_task_per_episode

        super().__init__(mdp_info)

    @property
    def env_names(self):
        return self._env_names
    
    @property
    def mdps(self):
        return self._training_envs

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError

        
    
class MetaWorldTaskEnv(Environment):
    def __init__(self, env, 
                        env_name,
                        observation_space,
                        action_space,
                        horizon=500, 
                        gamma=0.99, 
                        task_list = None,
                        normalize_reward = False, 
                        sample_task_per_episode=False):
        
        # MDP creation
        self._not_pybullet = True
        self._first = True
        if pybullet_found and '- ' + env_name in pybullet_envs.getList():
            import pybullet
            pybullet.connect(pybullet.DIRECT)
            self._not_pybullet = False

        self._env = env
        self._env_name = env_name
        self._task_list = task_list
        self._normalize_reward = normalize_reward
        self._sample_task_per_episode = sample_task_per_episode

        mdp_info = MDPInfo(observation_space=observation_space, action_space=action_space, gamma=gamma, horizon=horizon)

        super().__init__(mdp_info)


    def reset(self, state=None):
        if self._sample_task_per_episode and self._task_list is not None:
            task = random.choice(self._task_list)
            self._env.set_task(task)

        if state is None:
            # state, _ = self._env.reset()
            state = self._env.reset()
            return np.atleast_1d(state)
        else:
            self._env.reset()
            self._env.state = state
            return np.atleast_1d(state)
    
    def step(self, action):
        # action = self._convert_action(action)
        obs, reward, absorbing, info = self._env.step(action)
        return np.atleast_1d(obs), reward, absorbing, info
    

    def render(self):
        if self._first or self._not_pybullet:
            self._env.render() # ahmed: removed mode=mode argument from render function
            self._first = False
    
    def stop(self):
        try:
            if self._not_pybullet:
                self._env.close()
                self._env.viewer = None
                self._env._viewers = {}
        except:
            pass


    @property
    def env_name(self):
        return self._env_name