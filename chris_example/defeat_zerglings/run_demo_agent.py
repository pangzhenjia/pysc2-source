import sys

import gflags as flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env import run_loop

from chris_example.defeat_zerglings import demo_agent
from chris_example.maps import chris_maps

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 20000

FLAGS = flags.FLAGS


def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
            map_name="DefeatZerglingsAndBanelings",
            step_mul=step_mul,
            visualize=True,
            game_steps_per_episode=steps * step_mul) as env:
        demo_replay = []

        agent = demo_agent.MarineAgent(env=env)
        agent.env = env
        run_loop.run_loop([agent], env, steps)


if __name__ == '__main__':
    main()
