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
from pysc2.lib import point
from pysc2.lib import transform


from pysc2.my_agent.agent_network import ProbeNetwork

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PROBE_TYPE_INDEX = 84

_NO_OP = sc2_actions.FUNCTIONS.no_op.id

_MOVE_MINIMAP = sc2_actions.FUNCTIONS.Move_minimap.id
_BUILD_PYLON = sc2_actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_FORGE = sc2_actions.FUNCTIONS.Build_Forge_screen.id
_BUILD_CANNON = sc2_actions.FUNCTIONS.Build_PhotonCannon_screen.id

_ACTION_ARRAY = [_MOVE_MINIMAP, _BUILD_PYLON, _BUILD_FORGE, _BUILD_CANNON]
_ACTION_TYPE_NAME = ["move", "build_pylon", "build_forge", "build_cannon"]

_SELECT_POINT = sc2_actions.FUNCTIONS.select_point.id

_NOT_QUEUED = [0]


class SingleAgent(base_agent.BaseAgent):
    """My first agent for starcraft."""

    def __init__(self, env):
        super(SingleAgent, self).__init__()
        self.net = ProbeNetwork()
        self.env = env

        game_info = env.get_controller().game_info()
        self._map_size = point.Point.build(game_info.start_raw.map_size)
        fl_opts = game_info.options.feature_layer
        self._feature_layer_minimap_size = point.Point.build(fl_opts.minimap_resolution)
        self._features = features.Features(game_info)

        # define pos transform
        self._world_to_minimap = transform.Linear(point.Point(1, -1), point.Point(0, self._map_size.y))
        self._minimap_to_fl_minimap = transform.Linear(self._feature_layer_minimap_size / self._map_size)
        self._world_to_fl_minimap = transform.Chain(
            self._world_to_minimap,
            self._minimap_to_fl_minimap,
            transform.Floor()
        )

    def step(self, obs):
        super(SingleAgent, self).step(obs)

        map_data = obs.observation["minimap"][[0, 1, 5], :, :]

        # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon
        action_type_pro, action_type, pos = self.net.predict(map_data)

        print("predict: ", _ACTION_TYPE_NAME[action_type], "pos_x: ", pos.x, "pos_y: ", pos.y, action_type_pro)
        x = pos.x - 1
        y = pos.y - 1
        # if pos.y == 43 and action_type == 1:
        #     return sc2_actions.FunctionCall(_NO_OP, [])
        # if action_type == 2:
        #     y = 43

        if action_type == 4:
            return sc2_actions.FunctionCall(_NO_OP, [])

        if _ACTION_ARRAY[action_type] in obs.observation["available_actions"]:
            return sc2_actions.FunctionCall(_ACTION_ARRAY[action_type], [_NOT_QUEUED, [x, y]])
        else:
            return sc2_actions.FunctionCall(_NO_OP, [])

    def play(self, max_frames=0):
        total_frames = 0
        start_time = time.time()

        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()

        self.setup(observation_spec, action_spec)

        try:
            while True:

                # while loop to check start point
                while True:
                    timesteps = self.env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
                    visible_map = timesteps[0].observation["minimap"][1, :, :]
                    if visible_map[23, 17] == 0:
                        self.env.reset()
                    else:
                        break

                self.reset()

                # random select a probe
                unit_type_map = timesteps[0].observation["screen"][_UNIT_TYPE]
                pos_y, pos_x = (unit_type_map == _PROBE_TYPE_INDEX).nonzero()

                index = -10
                pos = [pos_x[index], pos_y[index]]
                timesteps = self.env.step([sc2_actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, pos])])

                for i in range(5):
                    self.env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])

                # single probe do some thing
                while True:
                    total_frames += 1
                    actions = [self.step(timesteps[0])]
                    if max_frames and total_frames >= max_frames:
                        return
                    if timesteps[0].last():
                        break
                    timesteps = self.env.step(actions)

        except KeyboardInterrupt:
            pass

        finally:
            elapsed_time = time.time() - start_time
            print("Took %.3f seconds for %s steps: %.3f fps" % (
                elapsed_time, total_frames, total_frames / elapsed_time))


