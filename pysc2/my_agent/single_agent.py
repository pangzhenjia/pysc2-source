# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

from pysc2.my_agent.utils import get_reward, test_probe
from pysc2.my_agent.agent_network import ProbeNetwork

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PROBE_TYPE_INDEX = 84

_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_SELECT_POINT = sc2_actions.FUNCTIONS.select_point.id
_MOVE_CAMERA = sc2_actions.FUNCTIONS.move_camera.id

_MOVE_MINIMAP = sc2_actions.FUNCTIONS.Move_minimap.id
_BUILD_PYLON = sc2_actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_FORGE = sc2_actions.FUNCTIONS.Build_Forge_screen.id
_BUILD_CANNON = sc2_actions.FUNCTIONS.Build_PhotonCannon_screen.id

_ACTION_ARRAY = [_MOVE_MINIMAP, _BUILD_PYLON, _BUILD_FORGE, _BUILD_CANNON]
_ACTION_TYPE_NAME = ["move", "build_pylon", "build_forge", "build_cannon"]

_NOT_QUEUED = [0]


class SingleAgent(base_agent.BaseAgent):
    """My first agent for starcraft."""

    def __init__(self, env):
        super(SingleAgent, self).__init__()
        self.net = ProbeNetwork()
        self.net.initialize()
        self.net.restore_rl_model()
        self.env = env

    def reset(self):
        super(SingleAgent, self).reset()

    def step(self, obs):
        super(SingleAgent, self).step(obs)

        # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon
        action_type_pro, action_type, x, y = self.net.predict(obs)

        print("predict: ", _ACTION_TYPE_NAME[action_type], "pos_x: ", x, "pos_y: ", y, action_type_pro)

        if _ACTION_ARRAY[action_type] in obs.observation["available_actions"]:
            return sc2_actions.FunctionCall(_ACTION_ARRAY[action_type], [_NOT_QUEUED, [x, y]])
        else:
            return sc2_actions.FunctionCall(_NO_OP, [])

    def play(self, max_frames=500):
        start_time = time.time()

        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()
        self.setup(observation_spec, action_spec)

        replay_buffer = []
        try:
            while True:

                self.reset()
                if self.episodes != 1:
                    self.env.reset()

                # while loop to check start point
                while True:
                    timesteps = self.env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
                    visible_map = timesteps[0].observation["minimap"][1, :, :]
                    if visible_map[23, 17] == 0:
                        self.env.reset()
                    else:
                        break

                # random select a probe
                unit_type_map = timesteps[0].observation["screen"][_UNIT_TYPE]
                pos_y, pos_x = (unit_type_map == _PROBE_TYPE_INDEX).nonzero()

                index = -5
                pos = [pos_x[index], pos_y[index]]
                timesteps = self.env.step([sc2_actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, pos])])

                # move camera
                timesteps = self.env.step([sc2_actions.FunctionCall(_MOVE_CAMERA, [[35, 47]])])

                # main loop
                total_frames = 0
                while True:
                    total_frames += 1
                    actions = [self.step(timesteps[0])]
                    last_timesteps = timesteps
                    timesteps = self.env.step(actions)

                    reward = timesteps[0].reward

                    if actions[0].function:
                        recoder = [last_timesteps[0], actions[0], timesteps[0], reward]
                        replay_buffer.append(recoder)

                    # probe_alive = test_probe(timesteps[0])
                    if not test_probe(timesteps[0]):
                        break

                    if max_frames and total_frames >= max_frames:
                        break
                    if timesteps[0].last():
                        break

                # some setting
                discount = 0.9
                learning_rate = 0.00001

                counter = self.episodes
                learning_rate = learning_rate * (1 - 0.9 * counter / max_frames)

                if counter % 100 == 0:
                    replay_buffer = []
                    self.net.update(replay_buffer, discount, learning_rate, counter)

        except KeyboardInterrupt:
            pass

        finally:
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


