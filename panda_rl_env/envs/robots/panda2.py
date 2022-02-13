import numpy as np
from gym import spaces

from panda_rl_env.envs.core import PyBulletRobot
from panda_rl_env.pybullet import PyBullet
from itertools import combinations
import contextlib

class Panda2(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """
    JOINT_LL = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
    JOINT_UL = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
    LINK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = True,
        base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        control_type: str = "ee",
    ) -> None:
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self._self_collision_check_list = self.get_self_collision_check_list()
        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        #ee_position = np.array(self.get_ee_position())
        #ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        # if not self.block_gripper:
        #     fingers_width = self.get_fingers_width()
        #     obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        # else:
        #     obs = np.concatenate((ee_position, ee_velocity))
        return self.get_joint_values()

    def get_joint_values(self):
        arm_joints = [0,1,2,3,4,5,6]
        joint_values = []
        for joint in arm_joints:
            joint_values.append(self.get_joint_angle(joint))
        return np.array(joint_values)

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def set_joint_random(self):
        """Set the robot to random pose."""
        for iter in range(100):
            rnd_joints = []
            for i in range(7):
                q = np.random.uniform(self.JOINT_LL[i], self.JOINT_UL[i])
                rnd_joints.append(q)
            self.sim.set_joint_angles(self.body_name, range(7), rnd_joints)
            if not self.is_self_collision():
                return np.array(rnd_joints)
        raise Exception("joint init fail")
    
    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self, joints=None) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        if joints is None:
            return self.get_link_position(self.ee_link)
        with self.preserving_joints(joints):
            return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def is_self_collision(self, joints=None):
        if joints is None:
            for linkA, linkB in self._self_collision_check_list:
                if self.sim.is_collision(self.body_name, self.body_name, linkA, linkB):
                    return True
            return False
        else:
            with self.preserving_joints(joints):
                for linkA, linkB in self._self_collision_check_list:
                    if self.sim.is_collision(self.body_name, self.body_name, linkA, linkB):
                        result = True
                result = False
            return result

    @contextlib.contextmanager
    def preserving_joints(self, joints):
        # Code to acquire resource, e.g.:
        joints_curr = self.get_joint_values()
        self.sim.set_joint_angles(self.body_name, range(7), joints)
        try:
            yield 
        finally:
            # Code to release resource, e.g.:
            self.sim.set_joint_angles(self.body_name, range(7), joints_curr)
    
    def get_self_collision_check_list(self):
        self_collision_check_list = list(combinations(self.LINK_INDICES, 2))
        self_collision_check_list = [pair for pair in self_collision_check_list if abs(pair[0]-pair[1]) != 1] #
        return self_collision_check_list