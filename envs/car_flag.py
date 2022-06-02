# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import Generator
import gym
from gym import spaces
from typing import Optional
from pyglet.canvas.xlib import NoSuchDisplayException

# Running on slurm
try:
    from gym.utils import pyglet_rendering as visualize
except NoSuchDisplayException:
    pass
except ImportError:
    pass


class CarFlag(gym.Env):
    def __init__(self, discrete: bool):
        self.max_position = 1.1
        self.min_position = -self.max_position
        self.max_speed = 0.07

        self.setup_view = False

        self.min_action = -1.0
        self.max_action = 1.0

        self.heaven_position = 1.0
        self.hell_position = -1.0
        self.priest_position = 0.5
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.screen_width = 600
        self.screen_height = 400

        # When the cart is within this vicinity, it observes the direction given
        # by the priest
        self.priest_delta = 0.2

        self.low_state = np.array(
            [self.min_position, -self.max_speed, -1.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed, 1.0], dtype=np.float32
        )

        world_width = self.max_position - self.min_position
        self.scale = self.screen_width / world_width

        self.discrete = discrete

        if self.discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
            )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, shape=(3,), dtype=np.float32
        )

        self.np_random = None

    def seed(self, seed):
        super().seed(seed)
        if self.np_random is None:
            seed_seq = np.random.SeedSequence(seed)
            self.np_random = Generator(np.random.PCG64(seed_seq))

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        if self.discrete:
            # 0 is -1, 1 is 0, 2 is 1
            force = action - 1
        else:
            force = np.clip(action, -1, 1)

        velocity += force * self.power
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        max_position = max(self.heaven_position, self.hell_position)
        min_position = min(self.heaven_position, self.hell_position)

        done = bool(position >= max_position or position <= min_position)

        env_reward = 0

        if self.heaven_position > self.hell_position:
            if position >= self.heaven_position:
                env_reward = 1.0

            if position <= self.hell_position:
                env_reward = -1.0

        if self.heaven_position < self.hell_position:
            if position <= self.heaven_position:
                env_reward = 1.0

            if position >= self.hell_position:
                env_reward = -1.0

        direction = 0.0
        if (
            position >= self.priest_position - self.priest_delta
            and position <= self.priest_position + self.priest_delta
        ):
            if self.heaven_position > self.hell_position:
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0

        self.state = np.array([position, velocity, direction])

        return self.state, env_reward, done, {"is_success": env_reward > 0}

    def render(self, mode="human"):
        self._setup_view()

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * self.scale, self._height(pos) * self.scale
        )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def reset(self):
        # Randomize the heaven/hell location
        if self.np_random.integers(low=0, high=2, size=1) == 0:
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0

        self.hell_position = -self.heaven_position

        if self.viewer is not None:
            self._draw_flags()
            self._draw_boundary()

        self.state = np.array([self.np_random.uniform(low=-0.2, high=0.2), 0, 0.0])
        return np.array(self.state)

    def _height(self, xs):
        return 0.55 * np.ones_like(xs)

    def _draw_boundary(self):
        flagx = (
            self.priest_position - self.priest_delta - self.min_position
        ) * self.scale
        flagy1 = self._height(self.priest_position) * self.scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

        flagx = (
            self.priest_position + self.priest_delta - self.min_position
        ) * self.scale
        flagy1 = self._height(self.priest_position) * self.scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

    def _draw_flags(self):
        scale = self.scale
        # Flag Heaven
        flagx = (abs(self.heaven_position) - self.min_position) * scale
        flagy1 = self._height(self.heaven_position) * scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # RED for hell
        if self.heaven_position > self.hell_position:
            flag.set_color(0.0, 1.0, 0)
        else:
            flag.set_color(1.0, 0.0, 0)

        self.viewer.add_geom(flag)

        # Flag Hell
        flagx = (-abs(self.heaven_position) - self.min_position) * scale
        flagy1 = self._height(self.hell_position) * scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # GREEN for heaven
        if self.heaven_position > self.hell_position:
            flag.set_color(1.0, 0.0, 0)
        else:
            flag.set_color(0.0, 1.0, 0)

        self.viewer.add_geom(flag)

        # BLUE for priest
        flagx = (self.priest_position - self.min_position) * scale
        flagy1 = self._height(self.priest_position) * scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.0, 0.0, 1.0)
        self.viewer.add_geom(flag)

    def _setup_view(self):
        if not self.setup_view:
            self.viewer = visualize.Viewer(self.screen_width, self.screen_height)
            scale = self.scale
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = visualize.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10
            carwidth = 40
            carheight = 20

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = visualize.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(visualize.Transform(translation=(0, clearance)))
            self.cartrans = visualize.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = visualize.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                visualize.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = visualize.make_circle(carheight / 2.5)
            backwheel.add_attr(
                visualize.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)

            self._draw_flags()
            self._draw_boundary()

            self.setup_view = True

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
