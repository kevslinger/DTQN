from gym.envs.registration import register
import gym


#######################
# DYNAMIC HEAVEN HELL #
#######################

register(
    id="DynamicHeavenHell-v0",
    entry_point="envs.dynamic_heaven_hell:DynamicHeavenHell",
    kwargs={},
    max_episode_steps=5000,
)

################
# MEMORY CARDS #
################

register(
    id="Memory-5-v0",
    entry_point="envs.memory_cards:Memory",
    kwargs={"num_pairs": 5},
    max_episode_steps=50,
)

############
# CAR FLAG #
############

register(
    id="DiscreteCarFlag-v0",
    entry_point="envs.car_flag:CarFlag",
    kwargs={"discrete": True},
    max_episode_steps=200,
)
