from __future__ import annotations

import numpy as np

from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle, Landmark
from highway_env.vehicle.graphics import VehicleGraphics


class ParallelParkingEnv(ParkingEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # horizontal size (left-right)
                "street_length": 60.0,
                # distance from center to each parking row
                "curb_offset": 10.0,
                # number of slots per side
                "n_slots": 8,
                # which bottom-row slot index is empty (0-based)
                "empty_slot_index": 3,  # can make this random later
                # wall thickness and margins
                "wall_margin": 4.0,
                "add_walls": True,
            }
        )
        return config

    # ------------------------------------------------------------------ #
    # ROAD + WALLS
    # ------------------------------------------------------------------ #
    def _create_road(self) -> None:
        net = RoadNetwork()
        L = self.config["street_length"]
        curb_offset = self.config["curb_offset"]
        wall_margin = self.config["wall_margin"]

        lane_width = 10.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)

        # Central driving lane (for ego): horizontal, y = 0
        net.add_lane(
            "drive_start",
            "drive_end",
            StraightLane(
                [0.0, 0.0],
                [L, 0.0],
                width=lane_width,
                line_types=lt,
            ),
        )

        # Dummy short vertical lanes -> white slot markers (top & bottom rows)
        n_slots = self.config["n_slots"]
        inner_margin_x = 5.0  # empty at left/right before first/after last slot
        total_slot_span = L - 2 * inner_margin_x
        slot_length = total_slot_span / n_slots

        for side, sign in (("bottom", -1.0), ("top", +1.0)):
            y_row = sign * curb_offset

            for i in range(n_slots + 1):
                x = inner_margin_x + i * slot_length
                # short vertical white line
                net.add_lane(
                    f"{side}_marker_{i}_in",
                    f"{side}_marker_{i}_out",
                    StraightLane(
                        [x, y_row - 2.0],   # 4m long vertical marker
                        [x, y_row + 2.0],
                        width=0.1,
                        line_types=lt,
                    ),
                )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        # Yellow rectangular border using four big obstacles
        if self.config["add_walls"]:
            top_y = curb_offset + wall_margin
            bot_y = -curb_offset - wall_margin
            left_x = 0.0
            right_x = L

            # top & bottom walls (horizontal)
            for y in (top_y, bot_y):
                wall = Obstacle(
                    self.road, [(left_x + right_x) / 2.0, y], heading=0.0)
                wall.LENGTH = (right_x - left_x) + 2.0
                wall.WIDTH = 1.0
                wall.diagonal = np.sqrt(wall.LENGTH**2 + wall.WIDTH**2)
                # make them yellow-ish
                wall.color = (1.0, 1.0, 0.0)
                self.road.objects.append(wall)

            # left & right walls (vertical)
            height = top_y - bot_y
            for x in (left_x, right_x):
                wall = Obstacle(
                    self.road, [x, (top_y + bot_y) / 2.0], heading=np.pi / 2.0)
                wall.LENGTH = height + 2.0
                wall.WIDTH = 1.0
                wall.diagonal = np.sqrt(wall.LENGTH**2 + wall.WIDTH**2)
                wall.color = (1.0, 1.0, 0.0)
                self.road.objects.append(wall)

    # ------------------------------------------------------------------ #
    # VEHICLES / PARKED CARS / GOAL
    # ------------------------------------------------------------------ #
    def _create_vehicles(self) -> None:
        n_slots = self.config["n_slots"]
        empty_idx = self.config["empty_slot_index"]
        curb_offset = self.config["curb_offset"]
        L = self.config["street_length"]

        self.controlled_vehicles = []

        # --- Ego in the central lane ---
        drive_lane = self.road.network.get_lane(
            ("drive_start", "drive_end", 0))
        ego_x = 0.2 * L
        ego_y = drive_lane.position(ego_x, 0)[1]

        ego = self.action_type.vehicle_class(
            self.road,
            [ego_x, ego_y],
            heading=0.0,
            speed=0.0,
        )
        ego.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        # Geometry for slot centers
        inner_margin_x = 5.0
        total_slot_span = L - 2 * inner_margin_x
        slot_length = total_slot_span / n_slots

        car_len = 4.5
        car_wid = 2.0

        # --- Top row: all slots filled ---
        for i in range(n_slots):
            x_center = inner_margin_x + (i + 0.5) * slot_length
            y_center = +curb_offset

            obstacle = Obstacle(self.road, [x_center, y_center], heading=0.0)
            obstacle.LENGTH = car_len
            obstacle.WIDTH = car_wid
            obstacle.diagonal = np.sqrt(car_len**2 + car_wid**2)
            self.road.objects.append(obstacle)

        # --- Bottom row: all filled except one empty slot (goal) ---
        goal_pos, goal_heading = None, None
        for i in range(n_slots):
            x_center = inner_margin_x + (i + 0.5) * slot_length
            y_center = -curb_offset

            if i == empty_idx:
                # empty slot -> goal here
                goal_pos = np.array([x_center, y_center])
                goal_heading = 0.0
                continue

            obstacle = Obstacle(self.road, [x_center, y_center], heading=0.0)
            obstacle.LENGTH = car_len
            obstacle.WIDTH = car_wid
            obstacle.diagonal = np.sqrt(car_len**2 + car_wid**2)
            self.road.objects.append(obstacle)

        # --- Goal landmark in the empty bottom slot ---
        ego.goal = Landmark(self.road, goal_pos, heading=goal_heading)
        self.road.objects.append(ego.goal)
