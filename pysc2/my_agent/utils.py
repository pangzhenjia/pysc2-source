import numpy as np


def get_relative_point(obs):
    _player_relative_index = 5
    _our_index = 1
    _enemy_index = 4

    _our_mul = 1
    _enemy_mul = 0.1

    data = obs.observation["minimap"][_player_relative_index, :, :]

    our_point = np.sum(data[32:, 32:] == _our_index) * _our_mul
    enemy_point = np.sum(data == _enemy_index) * _enemy_mul
    return our_point + enemy_point


def get_cannon_point(obs):
    # check how many cannons in screen
    _multiplier = 10

    # TODO: cannon


def get_reward(last_obs, now_obs):

    # reward rule: ( guess, still need to be done)
    #     1. enemy : each for 1 point
    #     2. our building in enemy highland:
    #         cannon: 10 point

    if now_obs.last():
        return 100

    last_obs_point = get_relative_point(last_obs)
    now_obs_point = get_relative_point(now_obs)
    reward = now_obs_point - last_obs_point

    return reward
