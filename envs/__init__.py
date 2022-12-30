from gym.envs.registration import register


try:
    import gym_gridverse
except ImportError:
    print(
        "WARNING: ``gym_gridverse`` is not installed. This means you cannot run an experiment with the gv_*.yaml domains."
    )

try:
    import gym_pomdps
except ImportError:
    print(
        "WARNING: ``gym_pomdps`` is not installed. This means you cannot run an experiment with the HeavenHell or "
        "Hallway domain. "
    )

try:
    import minihack
except ImportError:
    print(
        "WARNING: ``mini_hack`` is not installed. This means you cannot run an experiment with any of the MH- domains."
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


#############
# MINI HACK #
#############


register(
    id="MH-Room-5-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-5x5-v0"},
    max_episode_steps=100,
)

register(
    id="MH-Room-5-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-5x5-v0", "obs_crop": 3},
    max_episode_steps=100,
)

register(
    id="MH-DarkRoom-5-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-5x5-v0"},
    max_episode_steps=100,
)

register(
    id="MH-DarkRoom-5-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-5x5-v0", "obs_crop": 3},
    max_episode_steps=100,
)

register(
    id="MH-Room-15-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-15x15-v0"},
    max_episode_steps=300,
)

register(
    id="MH-Room-15-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-15x15-v0", "obs_crop": 3},
    max_episode_steps=300,
)

register(
    id="MH-DarkRoom-15-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-15x15-v0"},
    max_episode_steps=300,
)

register(
    id="MH-DarkRoom-15-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-Room-Dark-15x15-v0", "obs_crop": 3},
    max_episode_steps=300,
)

register(
    id="MH-Maze-9-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-9x9-v0"},
    max_episode_steps=180,
)

register(
    id="MH-Maze-9-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-9x9-v0", "obs_crop": 3},
    max_episode_steps=180,
)

register(
    id="MH-MazeMap-9-v0",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-Mapped-9x9-v0"},
    max_episode_steps=180,
)

register(
    id="MH-MazeMap-9-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": "MiniHack-MazeWalk-Mapped-9x9-v0", "obs_crop": 3},
    max_episode_steps=180,
)

des_maze_v0 = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
||||||||||||
|.|....|.|.|
|...||.|.|.|
||.|||...|.|
|..|...|||.|
||||.|||...|
|..........|
||||||||||||
ENDMAP
STAIR:(10, 1),down
BRANCH: (1,1,1,1),(2,2,2,2)
"""

register(
    id="MH-maze-v1",
    entry_point="envs.mini_hack:MiniHackWrapper",
    kwargs={"env_id": None, "obs_crop": 3, "des_file": des_maze_v0},
    max_episode_steps=180,
)
