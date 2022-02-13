import gym
import panda_rl_env

import numpy as np

class Node:
    def __init__(self, x):
        self.x = x
        self.parent = None

class Tree:
    def __init__(self, node_root):
        self.data = [node_root]
        self.root = node_root
    
    def add_node(self, node, parent):
        node.parent = parent
        self.data.append(node)
    
    def nearest(self, node):
        distances = []
        for node_tree in self.data:
            d = np.linalg.norm(node_tree.x - node.x)
            distances.append(d)
        idx = np.argmin(distances)
        return self.data[idx]
    
    def backtrack(self, node):
        path = [node.x]
        parent = node.parent
        while True:
            if parent is None:
                break
            path.append(parent.x)
            parent = parent.parent
        return path[::-1]

class RRT:
    def __init__(
        self, 
        node_start, 
        node_goal,
        is_collision):

        self.start = node_start
        self.goal = node_goal
        self.tree = Tree(self.start)
        self.is_collision = is_collision
        self.eps = 0.2
        self.is_goal = lambda node: np.linalg.norm(node.x - self.goal.x) < self.eps
        self.p_goal = 0.5

    def get_random_node(self):
        JOINT_LL = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
        JOINT_UL = [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671]
        rnd_joints = []
        for i in range(7):
            q = np.random.uniform(JOINT_LL[i], JOINT_UL[i])
            rnd_joints.append(q)
        return Node(np.array(rnd_joints))

    def control(self, node_near, node_rand):
        mag = np.linalg.norm(node_rand.x - node_near.x)
        if mag <= self.eps:
            node_new = node_rand
        else:
            joint_new = node_near.x + (node_rand.x - node_near.x) * self.eps
            node_new = Node(joint_new)

        if not self.is_collision(node_new.x):
            return node_new
        else:
            return None

    def extend(self, node_rand):
        node_near = self.tree.nearest(node_rand)
        node_new = self.control(node_near, node_rand)
        if node_new is not None:
            self.tree.add_node(node_new, node_near)
            if self.is_goal(node_new):
                self.last_node = node_new
                return "reached"
            return "advanced"
        return "trapped"

    def plan(self):
        for i in range(10000):
            if np.random.uniform(0,1) < self.p_goal:
                node_rand = self.goal
            else:
                node_rand = self.get_random_node()
            result = self.extend(node_rand)
            if result == "reached":
                break
        return self.tree.backtrack(self.last_node)

if __name__ == "__main__":
    env = gym.make("PandaReach2-v0",render=True)
    for i in range(10):
        obs = env.reset()
        action = env.action_space.sample()
        node_start = Node(obs["observation"])
        node_goal = Node(env.goal_joints)
        rrt = RRT(node_start, node_goal, env.env.robot.is_self_collision)
        path = rrt.plan()
        vels = []
        for i in range(len(path)-1):
            s1 = path[i]
            s2 = env.env.robot.get_ee_position(s1)
            s3 = env.env.task.get_goal()
            obs = {"observation":s1, "achieved_goal":s2, "desired_goal":s3}
            reward = env.env.task.compute_reward(
                obs["achieved_goal"], obs["desired_goal"], None)
            a = path[i+1] - path[i]
            print(s1, s2, s3, a, reward)


    input()