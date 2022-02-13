import os

from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaReach{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_rl_env.envs:PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPush{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_rl_env.envs:PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaSlide{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_rl_env.envs:PandaSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaPickAndPlace{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_rl_env.envs:PandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="PandaStack{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_rl_env.envs:PandaStackEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="PandaFlip{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_rl_env.envs:PandaFlipEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

register(
    id='PandaReach2-v0',
    entry_point='panda_rl_env.envs:PandaReachEnv2',
    max_episode_steps=100,
)