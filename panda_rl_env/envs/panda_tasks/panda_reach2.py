import numpy as np

from panda_rl_env.envs.core import RobotTaskEnv
from panda_rl_env.envs.robots.panda2 import Panda2
from panda_rl_env.envs.tasks.reach2 import Reach2
from panda_rl_env.pybullet import PyBullet
from typing import Any, Dict, Optional, Tuple, Union


class PandaReachEnv2(RobotTaskEnv):
    """Reach2 task wih Panda robot.

    observation : joints, current_ee_pose, goal_ee_pose
    action : joints 7
    reward : dense
    done : distance <= 0.05
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "dense", control_type: str = "ee", level=0.1) -> None:
        sim = PyBullet(render=render)
        robot = Panda2(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type="joints")
        task = Reach2(sim, reward_type="dense", get_ee_position=robot.get_ee_position)
        self.level = level
        super().__init__(robot, task)

    def reset(self) -> Dict[str, np.ndarray]:
        with self.sim.no_rendering():
            for i in range(100):
                joint_start = self.robot.set_joint_random()
                joint_rand = self.robot.set_joint_random()
                joint_goal = joint_start + (joint_rand - joint_start) * self.level
                self.sim.set_joint_angles(self.robot.body_name, range(7), joint_goal)
                if not self.robot.is_self_collision():
                    self.task.goal = np.array(self.robot.get_ee_position())
                    break
            self.sim.set_joint_angles(self.robot.body_name, range(7), joint_start)
            self.sim.set_base_pose("target", position=self.task.goal, orientation=np.array([0,0,0,1]))
        return self._get_obs()

    # def task_get_goal(self):
    #     pass
