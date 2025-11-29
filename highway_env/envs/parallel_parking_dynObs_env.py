from __future__ import annotations

from abc import abstractmethod

import numpy as np
from gymnasium import Env

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env needs this mapping
    PARKING_OBS = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": (-5.0, 5.0),
                    "speed_range": (-40, 40),
                },
                "reward_weights": [1, 1, 0.1, 0.1, 1, 1],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(60),
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 100,
                "screen_width": 600,
                "screen_height": 300,
                "centering_position": [0.5, 0.5],
                "scaling": 7,
                "controlled_vehicles": 1,
                "vehicles_count": 11,
                "add_walls": True,
                "manual_vehicle_position": None,
            }
        )
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(
                    agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(
                obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 6) -> None:
        """
        Create a road composed of straight adjacent lanes laid out for parallel parking.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        self.parking_spot_lanes = []
        width = 4.0
        lane_lt = (LineType.NONE, LineType.NONE)  # hide horizontal sidelines
        marker_lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 10
        slot_boundaries = set()

        for k in range(spots):
            # x coordinate of the *start* of each slot
            x = (k + 1 - spots // 2) * (length + x_offset) - length / 2

            # Top row: slots parallel to x-axis at y = +y_offset
            top_lane = StraightLane(
                [x, y_offset],          # start of slot
                [x + length, y_offset],  # end of slot (horizontal)
                width=width,
                line_types=lane_lt,
            )
            net.add_lane("a", "b", top_lane)
            self.parking_spot_lanes.append(
                ("a", "b", len(net.graph["a"]["b"]) - 1))

            # Bottom row: slots parallel to x-axis at y = -y_offset
            bottom_lane = StraightLane(
                [x, -y_offset],
                [x + length, -y_offset],
                width=width,
                line_types=lane_lt,
            )
            net.add_lane("b", "c", bottom_lane)
            self.parking_spot_lanes.append(
                ("b", "c", len(net.graph["b"]["c"]) - 1))

            # Collect boundaries for vertical markers
            slot_boundaries.update({x, x + length})

        # Bounds to align walls with the slot grid
        slot_min_x = min(slot_boundaries)
        slot_max_x = max(slot_boundaries)
        half_lane_width = width / 2
        margin_x = 2.0
        margin_y = 2.0
        self.wall_bounds = {
            "left_x": slot_min_x - margin_x,
            "right_x": slot_max_x + margin_x,
            "top_y": y_offset + half_lane_width + margin_y,
            "bot_y": -y_offset - half_lane_width - margin_y,
        }

        # Vertical markers at slot boundaries (purely for rendering)
        marker_width = 0.1
        for i, boundary_x in enumerate(sorted(slot_boundaries)):
            net.add_lane(
                f"top_marker_{i}_in",
                f"top_marker_{i}_out",
                StraightLane(
                    [boundary_x, y_offset - half_lane_width],
                    [boundary_x, y_offset + half_lane_width],
                    width=marker_width,
                    line_types=marker_lt,
                ),
            )
            net.add_lane(
                f"bottom_marker_{i}_in",
                f"bottom_marker_{i}_out",
                StraightLane(
                    [boundary_x, -y_offset - half_lane_width],
                    [boundary_x, -y_offset + half_lane_width],
                    width=marker_width,
                    line_types=marker_lt,
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Ego in center, two dynamic cars in center lane, plus parked cars."""

        # -----------------------------------
        # Geometry for the center lane region
        # -----------------------------------
        # Use wall_bounds to know the horizontal extent of the scene
        bounds = getattr(
            self,
            "wall_bounds",
            {"left_x": -35.0, "right_x": 35.0, "top_y": 21.0, "bot_y": -21.0},
        )
        left_x, right_x = bounds["left_x"], bounds["right_x"]
        center_x = (left_x + right_x) / 2.0
        center_y = 0.0

        # margin from the walls so cars aren't partly inside them
        margin = 3.0

        # -------------------------
        # 1. Ego vehicle (green car)
        # -------------------------
        self.controlled_vehicles = []

        ego = self.action_type.vehicle_class(
            self.road,
            [center_x, center_y],  # center of the middle lane
            heading=0.0,           # facing right
            speed=0.0,             # RL agent will control speed
        )
        ego.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        # -------------------------
        # 2. Dynamic vehicle 1: left → right (yellow)
        # -------------------------
        dyn1 = Vehicle(
            self.road,
            [left_x + margin, center_y - 5.0],  # near left boundary
            heading=0.0,                  # facing right
            speed=3.0,                    # 3 m/s
        )
        dyn1.LENGTH = 4.5
        dyn1.WIDTH = 2.0
        dyn1.diagonal = np.sqrt(dyn1.LENGTH**2 + dyn1.WIDTH**2)
        dyn1.color = VehicleGraphics.DEFAULT_COLOR
        self.road.vehicles.append(dyn1)

        # -------------------------
        # 3. Dynamic vehicle 2: right → left (yellow)
        # -------------------------
        dyn2 = Vehicle(
            self.road,
            [right_x - margin, center_y + 5.0],  # near right boundary
            heading=np.pi,                 # facing left
            speed=3.0,                     # 3 m/s
        )
        dyn2.LENGTH = 4.5
        dyn2.WIDTH = 2.0
        dyn2.diagonal = np.sqrt(dyn2.LENGTH**2 + dyn2.WIDTH**2)
        dyn2.color = VehicleGraphics.DEFAULT_COLOR
        self.road.vehicles.append(dyn2)

        # -----------------------------------
        # 4. Parallel-parked vehicles + goal
        # -----------------------------------
        # Only use the parking lanes for parked cars / goal
        empty_spots = list(
            getattr(self, "parking_spot_lanes",
                    self.road.network.lanes_dict().keys())
        )

        # Goal: pick one random parking slot as the goal for the ego
        if empty_spots:
            lane_index = empty_spots[self.np_random.choice(
                np.arange(len(empty_spots))
            )]
            lane = self.road.network.get_lane(lane_index)
            ego.goal = Landmark(
                self.road,
                lane.position(lane.length / 2, 0),  # center of slot
                heading=lane.heading,
            )
            self.road.objects.append(ego.goal)
            empty_spots.remove(lane_index)

        # Parked cars in remaining slots
        for _ in range(self.config["vehicles_count"]):
            if not empty_spots:
                break
            lane_index = empty_spots[self.np_random.choice(
                np.arange(len(empty_spots))
            )]
            lane = self.road.network.get_lane(lane_index)
            v = Vehicle.make_on_lane(
                self.road,
                lane_index,
                longitudinal=lane.length / 2,  # center of each slot
                speed=0.0,
            )
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # -------------------------
        # 5. Walls (unchanged)
        # -------------------------
        if self.config["add_walls"]:
            wall_thickness = 1.0
            top_y, bot_y = bounds["top_y"], bounds["bot_y"]
            cx = (left_x + right_x) / 2.0
            cy = (top_y + bot_y) / 2.0
            total_width = (right_x - left_x) + 2 * wall_thickness
            total_height = (top_y - bot_y) + 2 * wall_thickness

            # top & bottom walls (horizontal)
            for y in (top_y, bot_y):
                obstacle = Obstacle(self.road, [cx, y], heading=0.0)
                obstacle.LENGTH, obstacle.WIDTH = (total_width, wall_thickness)
                obstacle.diagonal = np.sqrt(
                    obstacle.LENGTH**2 + obstacle.WIDTH**2
                )
                self.road.objects.append(obstacle)

            # left & right walls (vertical)
            for x in (left_x, right_x):
                obstacle = Obstacle(
                    self.road, [x, cy], heading=np.pi / 2
                )
                obstacle.LENGTH, obstacle.WIDTH = (
                    total_height, wall_thickness
                )
                obstacle.diagonal = np.sqrt(
                    obstacle.LENGTH**2 + obstacle.WIDTH**2
                )
                self.road.objects.append(obstacle)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"],
                             agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10})
