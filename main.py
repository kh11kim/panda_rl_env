import numpy as np
import gym
import panda_rl_env
from rrt import Node, RRT
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

env = gym.make("PandaReach2-v0",render=True)
model = TD3("MultiInputPolicy", env, learning_rate=0.001,replay_buffer_kwargs={"handle_timeout_termination":False})

class RRTCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RRTCallback, self).__init__(verbose)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.pos_start = self.model.replay_buffer.pos

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.env = self.training_env
        buf = self.model.replay_buffer
        last_done = buf.dones[buf.pos]
        if last_done == False:
            node_start = Node(buf.observations["observation"][self.pos_start,0])
            node_goal = Node(self.env.goal_joints)
            rrt = RRT(node_start, node_goal, env.env.robot.is_self_collision)
            path = rrt.plan()
            self._goal = self.env.env.task.get_goal()
            for i in range(len(path)-1):
                obs = self.get_obs_i(path, i)
                obs_tp1 = self.get_obs_i(path, i+1)
                reward = self.env.env.task.compute_reward(
                    obs["achieved_goal"], obs["desired_goal"], None)
                done = 0 if i != len(path)-1 else 1
                a = path[i+1] - path[i]
                buf.add(obs, obs_tp1, a, np.array(reward), np.array(done), {"TimeLimit.truncated":False})

    def get_obs_i(self, path, i):
        s1 = path[i]
        s2 = self.env.env.robot.get_ee_position(s1)
        s3 = self._goal
        return {"observation":s1, "achieved_goal":s2, "desired_goal":s3}

callback = RRTCallback(env)
model.learn(10000, callback=callback)

# for i in range(100):
#     obs = env.reset()

# for i in range(100):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
input()