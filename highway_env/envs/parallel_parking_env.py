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

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
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
                    "acceleration_range": (-1.0, 1.0),
                    # "steering_range": np.deg2rad(45),  # ±60° instead of default ±45°
                    # "speed_range": (-1, 1),
                },
                "reward_weights": [1, 1, 0.1, 0.1, 1, 1],
                "success_goal_reward": 0.08,
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
                "vehicles_count": 24,
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
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 8) -> None:
        """
        Create a road composed of straight adjacent lanes laid out for parallel parking.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        self.parking_spot_lanes = []
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        slot_boundaries = set()

        for k in range(spots):
            # x coordinate of the *start* of each slot
            x = (k + 1 - spots // 2) * (length + x_offset) - length / 2

            # Top row: slots parallel to x-axis at y = +y_offset
            top_lane = StraightLane(
                [x, y_offset],          # start of slot
                [x + length, y_offset], # end of slot (horizontal)
                width=width,
                line_types=lt,
            )
            net.add_lane("a", "b", top_lane)
            self.parking_spot_lanes.append(("a", "b", len(net.graph["a"]["b"]) - 1))

            # Bottom row: slots parallel to x-axis at y = -y_offset
            bottom_lane = StraightLane(
                [x, -y_offset],
                [x + length, -y_offset],
                width=width,
                line_types=lt,
            )
            net.add_lane("b", "c", bottom_lane)
            self.parking_spot_lanes.append(("b", "c", len(net.graph["b"]["c"]) - 1))

            # Boundaries for vertical markers (shared between slots)
            slot_boundaries.update({x, x + length})

        # Store useful geometry for walls alignment
        slot_min_x = min(slot_boundaries)
        slot_max_x = max(slot_boundaries)
        margin_x = 2.0
        margin_y = 2.0
        half_lane_width = width / 2
        self.wall_bounds = {
            "left_x": slot_min_x - margin_x,
            "right_x": slot_max_x + margin_x,
            "top_y": y_offset + half_lane_width + margin_y,
            "bot_y": -y_offset - half_lane_width - margin_y,
        }

        # Vertical slot markers (short lanes used purely for rendering)
        marker_width = 0.1
        half_lane_width = width / 2
        for i, boundary_x in enumerate(sorted(slot_boundaries)):
            # Top row marker
            net.add_lane(
                f"top_marker_{i}_in",
                f"top_marker_{i}_out",
                StraightLane(
                    [boundary_x, y_offset - half_lane_width],
                    [boundary_x, y_offset + half_lane_width],
                    width=marker_width,
                    line_types=lt,
                ),
            )
            # Bottom row marker
            net.add_lane(
                f"bottom_marker_{i}_in",
                f"bottom_marker_{i}_out",
                StraightLane(
                    [boundary_x, -y_offset - half_lane_width],
                    [boundary_x, -y_offset + half_lane_width],
                    width=marker_width,
                    line_types=lt,
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # Only use the parking lanes for placement; ignore decorative marker lanes
        empty_spots = list(
            getattr(self, "parking_spot_lanes", self.road.network.lanes_dict().keys())
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = float(i - self.config["controlled_vehicles"] // 2) * 10.0
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0.0], 2.0 * np.pi * self.np_random.uniform(), 0.0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # Goal
        for vehicle in self.controlled_vehicles:
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # Other vehicles - New code for manual parked vehicle positioning
        # if self.config["manual_vehicle_positions"] is not None:
        #     # Manual positioning mode
        #     for position_config in self.config["manual_vehicle_positions"]:
        #         lane_index = position_config["lane_index"]
        #         longitudinal = position_config.get("longitudinal", 4.0)
        #         speed = position_config.get("speed", 0.0)
                
        #         # Convert string lane_index to tuple if needed
        #         if isinstance(lane_index, str):
        #             # Parse string like "('a', 'b', 0)" to tuple
        #             lane_index = eval(lane_index)
                
        #         if lane_index in empty_spots:
        #             v = Vehicle.make_on_lane(
        #                 self.road, lane_index, 
        #                 longitudinal=longitudinal, 
        #                 speed=speed
        #             )
        #             self.road.vehicles.append(v)
        #             empty_spots.remove(lane_index)
        # else:
        #     # Random positioning mode (original behavior)
        #     for i in range(self.config["vehicles_count"]):
        #         if not empty_spots:
        #             continue
        #         lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
        #         v = Vehicle.make_on_lane(self.road, lane_index, longitudinal=4.0, speed=0.0)
        #         self.road.vehicles.append(v)
        #         empty_spots.remove(lane_index)
        ### ORIGINAL CODE for Other Vehicles
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, longitudinal=4.0, speed=0.0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Walls
        if self.config["add_walls"]:
            bounds = getattr(
                self,
                "wall_bounds",
                {"left_x": -35, "right_x": 35, "top_y": 21, "bot_y": -21},
            )
            wall_thickness = 1.0
            left_x, right_x = bounds["left_x"], bounds["right_x"]
            top_y, bot_y = bounds["top_y"], bounds["bot_y"]
            center_x = (left_x + right_x) / 2
            center_y = (top_y + bot_y) / 2
            total_width = (right_x - left_x) + 2 * wall_thickness
            total_height = (top_y - bot_y) + 2 * wall_thickness

            # top & bottom walls (horizontal)
            for y in (top_y, bot_y):
                obstacle = Obstacle(self.road, [center_x, y], heading=0.0)
                obstacle.LENGTH, obstacle.WIDTH = (total_width, wall_thickness)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

            # left & right walls (vertical)
            for x in (left_x, right_x):
                obstacle = Obstacle(self.road, [x, center_y], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (total_height, wall_thickness)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
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
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
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
