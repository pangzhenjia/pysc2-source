import numpy as np
from pysc2.lib import features

_MINIMAP_SELECTED = features.MINIMAP_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PROBE_TYPE_INDEX = 84
_PYLON_TYPE_INDEX = 60
_FORGE_TYPE_INDEX = 63
_CANNON_TYPE_INDEX = 66


def test_probe(obs):
    return np.any(obs.observation["minimap"][_MINIMAP_SELECTED])


def calculate_reward(obs):
    unit_type_map = obs.observation["screen"][_UNIT_TYPE]

    pylon_mul = 1
    forge_mul = 5
    cannon_mul = 10

    reward = 0
    # reward += np.sum(unit_type_map == _PYLON_TYPE_INDEX) * pylon_mul
    reward += np.sum(unit_type_map == _FORGE_TYPE_INDEX) * forge_mul
    reward += np.sum(unit_type_map == _CANNON_TYPE_INDEX) * cannon_mul

    return reward


def get_reward(last_obs, now_obs):

    # reward rule: ( guess, still need to be done)
    #     1. pylon = 1
    #     2. forge = 5
    #     3. cannon = 10

    if now_obs.last():
        return 0

    last_obs_point = calculate_reward(last_obs)
    now_obs_point = calculate_reward(now_obs)
    reward = now_obs_point - last_obs_point - 1

    return reward


def pool_screen_power(power_map):

    pool_size = 4
    map_size = power_map.shape[0]

    out_size = map_size // pool_size
    out = np.zeros((out_size, out_size))

    for row_index in range(out_size):
        row_num = row_index * pool_size
        for col_index in range(out_size):
            col_num = col_index * pool_size
            out[row_index, col_index] = int(np.all(power_map[row_num:row_num+pool_size, col_num:col_num+4]))

    return out


def get_power_index(obs):

    minimap_camera = obs.observation["minimap"][3]
    screen_power = obs.observation["screen"][3]

    screen_unit_type = obs.observation["screen"][_UNIT_TYPE]
    screen_unit = (screen_unit_type == 0).astype("int")

    reduce_screen_power = np.logical_and(screen_power, screen_unit).astype("int")
    trans_power = pool_screen_power(reduce_screen_power).reshape(-1)

    minimap_camera = minimap_camera.reshape(-1)
    minimap_camera[minimap_camera == 1] = trans_power

    return minimap_camera == 1




