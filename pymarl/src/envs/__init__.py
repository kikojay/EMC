from functools import partial
# do not import SC2 in labtop
import socket
if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname():
    from smac.env import MultiAgentEnv, StarCraft2Env, Matrix_game1Env, Matrix_game2Env, Matrix_game3Env, mmdp_game1Env
else:
    from .multiagentenv import MultiAgentEnv
import sys
import os
from .stag_hunt import StagHunt
from .GridworldEnv import GridworldEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "matrix_game_1": partial(env_fn, env=Matrix_game1Env),
    "matrix_game_2": partial(env_fn, env=Matrix_game2Env),
    "matrix_game_3": partial(env_fn, env=Matrix_game3Env),
    "mmdp_game_1": partial(env_fn, env=mmdp_game1Env)
} if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname() else {}
REGISTRY["gridworld"] = GridworldEnv



REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
