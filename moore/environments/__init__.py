try:
    from .minigrid_env import MiniGrid
except:
    pass

# vectorized environment
from .base_vec_env import VecEnv, CloudpickleWrapper
from .subproc_vec_env import SubprocVecEnv