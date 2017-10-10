import os
import numpy as np
import glob

name_array = ["single_select",
              "multi_select",
              "build_queue",
              "cargo",
              "cargo_slots_available",
              "screen",
              "minimap",
              "game_loop",
              "score_cumulative",
              "player",
              "control_groups",
              "available_actions"
              ]


def save(path, obs, frame_num):
    assert os.path.exists(path)

    path = path + "/" + str(frame_num)
    if not os.path.exists(path):
        os.mkdir(path)

    for file_name in name_array:
        save_name = path + "/" + file_name + ".npy"
        np.save(save_name, obs[file_name])





