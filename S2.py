# Simulation 1: Office buidling, 1 layer

import pygame
import numpy as np
import random
import time
import math
import heapq
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from functools import lru_cache

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    # Grid and Display - DYNAMIC SCALING
    GRID_WIDTH = 48
    GRID_HEIGHT = 36
    TARGET_WINDOW_HEIGHT = 900
    TARGET_WINDOW_WIDTH = 1600
    
    # Calculate cell size dynamically to fit window
    CELL_SIZE = min(
        TARGET_WINDOW_HEIGHT // GRID_HEIGHT,
        (TARGET_WINDOW_WIDTH - 450) // GRID_WIDTH  # Reserve 450px for panel
    )
    
    # Simulation timing - CHANGED: 1 tick = 6 seconds
    TICK_RATE = 60
    SECONDS_PER_HAZARD_TICK = 6.0  # Each hazard tick = 6 seconds (was 3.0)
    TIME_SCALE = 3.0
    MAX_SIMULATION_TIME = 600
    
    RESPONSE_TIME = 0
        
    # Responder parameters
    V1_UNLOADED = 2.0
    V2_WALKING = 1.1
    V3_INJURED = 0.67
    P_STAIRS = 0.82
    MAX_CARRYING_CAPACITY = 1
    
    # Hazard thresholds
    DOOR_BURN_THRESHOLD = 0.3
    SMOKE_HAZARD_THRESHOLD = 0.3
    
    # Room search parameters
    SIGHT_LENGTH = 100
    HEADING_COUNT = 36
    
    # Priority system
    KAPPA = 0.1
    ALPHA_H = 0.75
    BETA_H = 0.75
    GAMMA_H = 0.4
    
    # Occupant distribution
    INJURED_RATIO_MIN = 0.1
    INJURED_RATIO_MAX = 0.4
    
    # Monte Carlo simulation
    MONTE_CARLO_RUNS = 5
    MONTE_CARLO_MAX_TICKS = 500
    
    # Dynamic font sizes based on cell size
    @staticmethod
    def get_font_size(base_size: int) -> int:
        """Scale font size based on cell size"""
        scale_factor = Config.CELL_SIZE / 30.0  # 30 was original cell size
        return max(10, int(base_size * scale_factor))
    
    # Colors (unchanged)
    COLORS = {
        0: (240, 240, 240),
        1: (200, 200, 100),
        2: (200, 150, 100),
        3: (150, 100, 50),
        6: (34, 139, 34),
        7: (135, 206, 235),
        8: (100, 100, 200),
        9: (50, 50, 70),
        'fire': (255, 0, 0),
        'smoke': (100, 100, 100),
        'responder': (255, 165, 0),
        'occupant_walking': (0, 255, 0),
        'occupant_injured': (255, 0, 255),
        'path': (255, 215, 0),
        'search_path': (0, 255, 255),
        'evac_path': (0, 255, 0),
        'waiting': (200, 200, 0)
    }

# Cell types
WALL = 9
WINDOW = 7
DOOR = 8
EXIT = 6
EMPTY = 0
LIGHT_OBSTACLE = 1
MEDIUM_OBSTACLE = 2
HEAVY_OBSTACLE = 3

WALKABLE_TILES = {EMPTY, LIGHT_OBSTACLE, MEDIUM_OBSTACLE, DOOR, EXIT}
IMPASSABLE_TILES = {WALL, HEAVY_OBSTACLE}
OBSTACLES = {LIGHT_OBSTACLE, MEDIUM_OBSTACLE, HEAVY_OBSTACLE}

NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# State enumerations
class ResponderState(Enum):
    IDLE = "idle"
    MOVING_TO_ROOM = "moving"
    SEARCHING_ROOM = "searching"
    EVACUATING = "evacuating"

class OccupantState(Enum):
    WAITING = "waiting"
    FOLLOWING = "following"
    CARRIED = "carried"
    RESCUED = "rescued"

class RoomStatus(Enum):
    UNVISITED = "unvisited"
    ASSIGNED = "assigned"
    SEARCHING = "searching"
    PARTIALLY_CLEARED = "partial"
    FULLY_CLEARED = "cleared"

# Data structures
@dataclass
class Occupant:
    id: int
    room_id: int
    position: Tuple[float, float]
    is_injured: bool
    state: OccupantState = OccupantState.WAITING
    assigned_responder: Optional[int] = None
    
@dataclass
class Room:
    id: int
    cells: Set[Tuple[int, int]]
    center: Tuple[int, int]
    area: float
    occupants: List[int]
    injured_count: int
    walking_count: int
    adjacent_doors: Set[Tuple[int, int]]
    status: RoomStatus = RoomStatus.UNVISITED
    assigned_responder: Optional[int] = None
    time_until_danger: float = float('inf')
    has_energy_control: bool = False
    has_hazardous_materials: bool = False
    has_confined_space: bool = False
    complexity_factor: float = 1.0
    priority: float = 0.0
    has_been_searched: bool = False  # NEW: Track if room was already searched

class Responder:
    def __init__(self, responder_id: int, start_exit: Tuple[int, int]):
        self.id = responder_id
        self.position = (float(start_exit[0]), float(start_exit[1]))
        self.state = ResponderState.IDLE
        self.carried_injured: List[int] = []
        self.walking_followers: List[int] = []
        self.assigned_room: Optional[int] = None
        self.current_path: List[Tuple[int, int]] = []
        self.path_index: int = 0
        self.search_path: List[Tuple[int, int]] = []
        self.search_path_index: int = 0
        self.current_trip: int = 1
        self.total_distance_traveled: float = 0
        self.total_occupants_rescued: int = 0
        self.total_rooms_cleared: int = 0
        
        # ISSUE 2: Track if responder needs to fill capacity
        self.needs_wounded: bool = False  # True if has walking but no injured
        
    def get_current_load(self) -> int:
        return len(self.carried_injured)
    
    def has_walking_followers(self) -> bool:
        return len(self.walking_followers) > 0
    
    def has_any_occupants(self) -> bool:
        """Check if carrying any occupants"""
        return len(self.carried_injured) > 0 or len(self.walking_followers) > 0
    
    def get_speed(self) -> float:
        l = self.get_current_load()
        w = 1 if self.has_walking_followers() else 0
        if l == 0 and w == 0:
            return Config.V1_UNLOADED
        elif l == 0 and w == 1:
            return Config.V2_WALKING
        else:
            return Config.V3_INJURED
    
    def at_capacity(self) -> bool:
        return len(self.carried_injured) >= Config.MAX_CARRYING_CAPACITY
    
    def should_continue_searching(self) -> bool:
        return len(self.walking_followers) > 0 and len(self.carried_injured) == 0
# ============================================================================
# FLOOR PLAN MANAGEMENT
# ============================================================================

class FloorPlan:
    """Manages the building floor plan and spatial relationships"""
    
    def __init__(self):
        self.grid = self._generate_floor_plan()
        self.height, self.width = self.grid.shape
        self.rooms: Dict[int, Room] = {}
        self.exits: Set[Tuple[int, int]] = set()
        self.doors: Set[Tuple[int, int]] = set()
        self.windows: Set[Tuple[int, int]] = set()
        
        self._identify_special_tiles()
        self._identify_rooms()
        self._calculate_room_properties()
        
    def _generate_floor_plan(self) -> np.ndarray:
        plan = np.array([
            [9,9,9,9,9,9,9,9,9,6,6,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,0,0,0,9,0,0,9,3,0,0,0,0,0,0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,1,2,1,9,0,0,9,3,0,0,3,0,3,0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,0,0,0,8,0,0,9,3,0,0,3,0,3,0,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,9,9,9,9,0,0,9,3,0,0,0,0,0,0,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,0,0,0,9,0,0,9,3,0,0,0,0,0,0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,1,2,1,9,0,0,9,9,0,0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,1,2,1,9,0,0,0,0,0,0,8,0,0,0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,0,0,0,8,0,0,0,0,0,0,9,0,0,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,9,9,9,9,9,9,9,9,0,0,9,8,8,8,9,9,9,9,9,6,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,3,3,3,3,9,3,3,9,0,0,9,1,0,1,8,0,0,0,9,0,0,9,0,3,0,9,0,3,0,9,0,0,0,0,9],
            [9,3,0,0,0,8,0,0,8,0,0,9,0,0,0,9,0,0,1,9,0,0,9,0,3,0,0,0,3,0,9,0,0,0,0,7],
            [9,3,0,0,0,8,0,0,8,0,0,9,1,0,1,9,9,9,9,9,0,0,9,1,3,1,9,1,3,1,9,0,0,0,0,7],
            [9,3,0,0,3,9,1,1,9,0,0,9,0,0,0,8,0,0,0,9,0,0,9,0,0,0,9,0,0,0,9,0,0,0,0,7],
            [9,3,3,3,3,9,3,3,9,0,0,9,1,1,1,9,0,0,1,9,0,0,9,2,0,0,9,0,0,2,9,0,0,0,0,9],
            [9,8,9,9,9,9,9,9,9,0,0,9,9,9,9,9,9,9,9,9,0,0,9,9,9,8,9,8,9,9,9,9,8,8,9,9],
            [9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9],
            [9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9],
            [9,9,9,9,9,0,0,0,0,0,0,9,9,8,9,9,8,8,9,9,9,9,9,9,9,9,8,9,9,9,9,8,9,9,9,9],
            [9,3,3,0,8,0,0,0,0,0,0,9,3,0,3,9,0,0,1,1,1,0,9,2,1,0,0,0,1,2,9,0,0,0,3,9],
            [9,0,0,0,9,1,0,0,0,0,1,9,3,0,3,9,1,2,2,2,2,1,9,2,1,0,0,0,1,2,9,0,0,0,3,9],
            [7,0,2,0,9,1,0,0,0,0,1,9,3,0,3,9,1,2,2,2,2,1,9,2,1,0,0,0,1,2,9,0,2,2,0,9],
            [7,0,1,0,9,1,0,0,0,0,1,9,3,0,3,9,0,1,1,1,1,0,9,2,1,0,0,0,1,2,9,0,1,1,0,9],
            [9,9,9,9,9,9,9,6,6,9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,7,7,7,7,7,9,9,9,7,7,9,9]
        ], dtype=int)
        return plan
    
    def _identify_special_tiles(self):
        """Identify exits, doors, and windows"""
        for i in range(self.height):
            for j in range(self.width):
                cell_type = int(self.grid[i, j])
                if cell_type == EXIT:
                    self.exits.add((i, j))
                elif cell_type == DOOR:
                    self.doors.add((i, j))
                elif cell_type == WINDOW:
                    self.windows.add((i, j))
    
    def _identify_rooms(self):
        """Identify rooms using flood-fill algorithm"""
        visited = np.zeros((self.height, self.width), dtype=bool)
        room_id = 0
        
        for i in range(self.height):
            for j in range(self.width):
                cell_type = int(self.grid[i, j])
                if visited[i, j] or cell_type in IMPASSABLE_TILES or cell_type in {WINDOW, EXIT}:
                    continue
                if cell_type == DOOR:
                    continue
                
                # Flood fill to find room
                cells = self._flood_fill(i, j, visited)
                if cells:
                    self.rooms[room_id] = Room(
                        id=room_id,
                        cells=cells,
                        center=self._calculate_center(cells),
                        area=len(cells),  # Will be converted to m² later
                        occupants=[],
                        injured_count=0,
                        walking_count=0,
                        adjacent_doors=self._find_adjacent_doors(cells)
                    )
                    room_id += 1
    
    def _flood_fill(self, start_i: int, start_j: int, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """Flood fill to identify connected room cells"""
        queue = deque([(start_i, start_j)])
        cells = set()
        
        while queue:
            i, j = queue.popleft()
            if visited[i, j]:
                continue
            
            visited[i, j] = True
            cells.add((i, j))
            
            # Check 4-connected neighbors
            for di, dj in NEI4:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    continue
                if visited[ni, nj]:
                    continue
                
                neighbor_type = int(self.grid[ni, nj])
                if neighbor_type in IMPASSABLE_TILES or neighbor_type in {WINDOW, EXIT, DOOR}:
                    continue
                
                queue.append((ni, nj))
        
        return cells
    
    def _calculate_center(self, cells: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate centroid of room cells"""
        if not cells:
            return (0, 0)
        total_i = sum(i for i, j in cells)
        total_j = sum(j for i, j in cells)
        count = len(cells)
        return (total_i // count, total_j // count)
    
    def _find_adjacent_doors(self, cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Find all doors adjacent to room cells"""
        adjacent_doors = set()
        for i, j in cells:
            for di, dj in NEI4:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    if int(self.grid[ni, nj]) == DOOR:
                        adjacent_doors.add((ni, nj))
        return adjacent_doors
    
    def _calculate_room_properties(self):
        """Calculate room areas and complexity factors"""
        for room in self.rooms.values():
            # Area in m² (assuming each cell is 1m x 1m)
            room.area = float(len(room.cells))
            
            # Assign complexity factors (simplified - can be customized per room)
            if random.random() < 0.2:
                room.has_energy_control = True
            if random.random() < 0.15:
                room.has_hazardous_materials = True
            if random.random() < 0.1:
                room.has_confined_space = True
            
            # Calculate H_i (Equation 4)
            F1 = 1 if room.has_energy_control else 0
            F2 = 1 if room.has_hazardous_materials else 0
            F3 = 1 if room.has_confined_space else 0
            
            C1, C2, C3 = 0.6, 0.8, 1.0
            room.complexity_factor = 1.0 + Config.GAMMA_H * (F1*C1 + F2*C2 + F3*C3)
    
    def get_room_for_cell(self, i: int, j: int) -> Optional[int]:
        """Get room ID for a given cell"""
        for room_id, room in self.rooms.items():
            if (i, j) in room.cells:
                return room_id
        return None
    
    def is_walkable(self, i: int, j: int) -> bool:
        """Check if cell is walkable"""
        if not (0 <= i < self.height and 0 <= j < self.width):
            return False
        return int(self.grid[i, j]) in WALKABLE_TILES
# ============================================================================
# HAZARD SIMULATION (FIXED D_i CALCULATION)
# ============================================================================

class HazardSimulation:
    """Fire and smoke propagation - EXACT logic from hazard_v3.py"""
    
    def __init__(self, floor_plan: 'FloorPlan'):
        self.floor_plan = floor_plan
        self.height = floor_plan.height
        self.width = floor_plan.width
        
        # Hazard state arrays
        self.fire_intensity = np.zeros((self.height, self.width), dtype=float)
        self.smoke_intensity = np.zeros((self.height, self.width), dtype=float)
        self.fuel_remaining = np.zeros((self.height, self.width), dtype=float)
        self.fuel_initial = np.zeros((self.height, self.width), dtype=float)
        self.door_burned = np.zeros((self.height, self.width), dtype=bool)
        
        # Tracking
        self.fire_ticks = np.full((self.height, self.width), -1, dtype=int)
        self.smoke_ticks = np.full((self.height, self.width), -1, dtype=int)
        self.current_tick = 0
        
        # Fuel properties (EXACT from hazard_v3.py)
        self.fuel_properties = {
            EMPTY: {"max_intensity": 0.6, "burn_rate": 0.10, "fuel_load": 10.0, "smoke_yield": 0.3},
            LIGHT_OBSTACLE: {"max_intensity": 0.9, "burn_rate": 0.25, "fuel_load": 15.0, "smoke_yield": 0.6},
            MEDIUM_OBSTACLE: {"max_intensity": 1.0, "burn_rate": 0.20, "fuel_load": 30.0, "smoke_yield": 1.0},
            HEAVY_OBSTACLE: {"max_intensity": 1.0, "burn_rate": 0.15, "fuel_load": 70.0, "smoke_yield": 1.8},
            DOOR: {"max_intensity": 0.7, "burn_rate": 0.20, "fuel_load": 30.0, "smoke_yield": 0.8},
            WINDOW: {"max_intensity": 0.0, "burn_rate": 0.0, "fuel_load": 0.0, "smoke_yield": 0.0}
        }
        
        # Fire spread probabilities (EXACT from hazard_v3.py)
        self.fire_spread_base_prob = {
            EMPTY: 0.15,
            LIGHT_OBSTACLE: 0.40,
            MEDIUM_OBSTACLE: 0.65,
            HEAVY_OBSTACLE: 0.80,
            DOOR: 0.25
        }
        
        self._initialize_fuel()
        self.room_D_i: Dict[int, float] = {}
        
    def _initialize_fuel(self):
        """Initialize fuel loads"""
        for i in range(self.height):
            for j in range(self.width):
                cell_type = int(self.floor_plan.grid[i, j])
                props = self.fuel_properties.get(cell_type, self.fuel_properties[EMPTY])
                self.fuel_remaining[i, j] = props["fuel_load"]
                self.fuel_initial[i, j] = props["fuel_load"]
    
    def ignite_fire(self, i: int, j: int, intensity: float = 0.5):
        """Start fire at location"""
        if 0 <= i < self.height and 0 <= j < self.width:
            cell_type = int(self.floor_plan.grid[i, j])
            if cell_type not in {WALL, WINDOW}:
                self.fire_intensity[i, j] = intensity
                self.fire_ticks[i, j] = 0
                print(f"[HAZARD] Fire ignited at ({i}, {j})")
    
    def run_monte_carlo_time_to_danger(self, start_i: int, start_j: int):
        """
        Monte Carlo simulation for D_i calculation
        FIXED: Proper hazard conditions from essay Section 2.3.2
        """
        runs = Config.MONTE_CARLO_RUNS
        max_ticks = Config.MONTE_CARLO_MAX_TICKS
        
        print(f"\n[MONTE CARLO] Running {runs} simulations for D_i calculation...")
        
        # Save current state
        saved_state = {
            "fire": self.fire_intensity.copy(),
            "smoke": self.smoke_intensity.copy(),
            "fuel": self.fuel_remaining.copy(),
            "door_burned": self.door_burned.copy(),
            "tick": self.current_tick,
            "fire_ticks": self.fire_ticks.copy(),
            "smoke_ticks": self.smoke_ticks.copy()
        }
        
        # Get room info
        rooms = {rid: room.cells for rid, room in self.floor_plan.rooms.items()}
        cell_to_room = {c: rid for rid, cells in rooms.items() for c in cells}
        
        # Track adjacent doors
        room_doors = {rid: set() for rid in rooms}
        for rid, cells in rooms.items():
            for ci, cj in cells:
                for di, dj in NEI4:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < self.height and 0 <= nj < self.width:
                        if int(self.floor_plan.grid[ni, nj]) == DOOR:
                            room_doors[rid].add((ni, nj))
        
        per_room_times = {rid: [] for rid in rooms}
        exits = {(i, j) for i in range(self.height) for j in range(self.width)
                if int(self.floor_plan.grid[i, j]) == EXIT}
        
        base_seed = int(time.time() * 1000) & 0xffffffff
        
        for run in range(runs):
            random.seed(base_seed ^ (run * 7919))
            
            # Reset to initial state
            self.fire_intensity = saved_state["fire"].copy()
            self.smoke_intensity = saved_state["smoke"].copy()
            self.fuel_remaining = saved_state["fuel"].copy()
            self.door_burned = saved_state["door_burned"].copy()
            self.current_tick = 0
            self.fire_ticks = saved_state["fire_ticks"].copy()
            self.smoke_ticks = saved_state["smoke_ticks"].copy()
            
            # Ignite fire
            self.ignite_fire(start_i, start_j, 0.5)
            
            room_hazard_tick = {rid: None for rid in rooms}
            start_room = cell_to_room.get((start_i, start_j))
            if start_room is not None:
                room_hazard_tick[start_room] = 0
            
            # Simulate hazard spread
            for t in range(max_ticks):
                self.spread_hazard()
                
                # Check hazard conditions for each room (FROM ESSAY)
                for rid, cells in rooms.items():
                    if room_hazard_tick[rid] is not None:
                        continue
                    
                    # Condition 1: Direct fire in room
                    has_fire = any(self.fire_intensity[ci, cj] > 0 for ci, cj in cells)
                    if has_fire:
                        room_hazard_tick[rid] = t
                        continue
                    
                    # Condition 2: Fire at door (blocking exit)
                    door_fire = any(self.fire_intensity[di, dj] > Config.DOOR_BURN_THRESHOLD
                                   for di, dj in room_doors[rid])
                    if door_fire:
                        room_hazard_tick[rid] = t
                        continue
                    
                    # Condition 3: Critical smoke density
                    avg_smoke = sum(self.smoke_intensity[ci, cj] for ci, cj in cells) / len(cells)
                    if avg_smoke >= Config.SMOKE_HAZARD_THRESHOLD:
                        room_hazard_tick[rid] = t
                        continue
                    
                    # Condition 4: No safe path to exit (BFS through non-fire cells)
                    has_path = self._check_safe_path_exists(cells, exits)
                    if not has_path:
                        room_hazard_tick[rid] = t
                        continue
                
                # Stop if all rooms hazardous
                if all(room_hazard_tick[rid] is not None for rid in rooms):
                    break
            
            # Record times
            for rid in rooms:
                tick = room_hazard_tick[rid] if room_hazard_tick[rid] is not None else max_ticks
                per_room_times[rid].append(tick)
        
        # Restore state
        self.fire_intensity = saved_state["fire"].copy()
        self.smoke_intensity = saved_state["smoke"].copy()
        self.fuel_remaining = saved_state["fuel"].copy()
        self.door_burned = saved_state["door_burned"].copy()
        self.current_tick = saved_state["tick"]
        self.fire_ticks = saved_state["fire_ticks"].copy()
        self.smoke_ticks = saved_state["smoke_ticks"].copy()
        
        # Calculate statistics and store D_i
        stats = {}
        for rid, times in per_room_times.items():
            arr = np.array(times, dtype=float)
            p90 = float(np.percentile(arr, 90))
            stats[rid] = {
                "min": int(np.min(arr)),
                "p90": p90,
                "mean": float(np.mean(arr)),
                "max": int(np.max(arr))
            }
            # Store p90 as D_i (time until danger)
            self.room_D_i[rid] = p90
        
        print("\n[MONTE CARLO] Time-until-danger (D_i) results:")
        for rid, s in stats.items():
            d_i = self.room_D_i[rid]
            print(f"  Room {rid}: D_i={d_i:.1f} (min={s['min']}, p90={s['p90']:.1f}, mean={s['mean']:.1f}, max={s['max']})")
        
        self._export_visualization(stats, start_i, start_j)
        return stats
    
    def _check_safe_path_exists(self, room_cells: Set[Tuple[int, int]], 
                                exits: Set[Tuple[int, int]]) -> bool:
        """BFS to check if safe path exists from room to any exit"""
        if not exits or not room_cells:
            return False
        
        visited = set()
        queue = deque()
        
        # Start from exits (reverse search)
        for ex in exits:
            queue.append(ex)
            visited.add(ex)
        
        while queue:
            ci, cj = queue.popleft()
            
            # Found path to room
            if (ci, cj) in room_cells:
                return True
            
            for di, dj in NEI4:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    continue
                if (ni, nj) in visited:
                    continue
                
                cell_type = int(self.floor_plan.grid[ni, nj])
                if cell_type in {WALL, WINDOW}:
                    continue
                
                # Cannot pass through fire
                if self.fire_intensity[ni, nj] > 0:
                    continue
                
                visited.add((ni, nj))
                queue.append((ni, nj))
        
        return False
    
    def spread_hazard(self):
        """Advance hazard by one tick (EXACT from hazard_v3.py)"""
        self.current_tick += 1
        
        new_fire = self.fire_intensity.copy()
        new_smoke = self.smoke_intensity.copy()
        
        # Door burning
        for i, j in self.floor_plan.doors:
            if self.fire_intensity[i, j] > Config.DOOR_BURN_THRESHOLD:
                self.door_burned[i, j] = True
        
        # Fire growth and spread (EXACT logic from hazard_v3.py)
        for i in range(self.height):
            for j in range(self.width):
                if self.fire_intensity[i, j] > 0:
                    cell_type = int(self.floor_plan.grid[i, j])
                    props = self.fuel_properties.get(cell_type, self.fuel_properties[EMPTY])
                    
                    max_int = props["max_intensity"]
                    burn_rate = props["burn_rate"]
                    
                    # Growth
                    if max_int > 0 and self.fuel_remaining[i, j] > 0:
                        growth = burn_rate * self.fire_intensity[i, j] * (1 - self.fire_intensity[i, j] / max_int)
                        new_fire[i, j] = min(max_int, self.fire_intensity[i, j] + growth)
                    else:
                        new_fire[i, j] = max(0.0, self.fire_intensity[i, j] - 0.12)
                    
                    # Fuel consumption
                    fuel_cons = min(burn_rate * self.fire_intensity[i, j] * 0.9, self.fuel_remaining[i, j])
                    self.fuel_remaining[i, j] -= fuel_cons
                    
                    # Smoke production
                    if self.fuel_initial[i, j] > 0:
                        smoke_prod = (fuel_cons / self.fuel_initial[i, j]) * props["smoke_yield"] * 25.0
                        new_smoke[i, j] = min(1.0, new_smoke[i, j] + min(smoke_prod, 0.6))
                    
                    # Fire spread
                    for di, dj in NEI8:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < self.height and 0 <= nj < self.width):
                            continue
                        
                        nt = int(self.floor_plan.grid[ni, nj])
                        if nt in {WALL, WINDOW}:
                            continue
                        
                        if self.fire_intensity[ni, nj] == 0:
                            base_prob = self.fire_spread_base_prob.get(nt, 0.0)
                            dist_factor = 1.0 / (1.0 + math.hypot(di, dj))
                            
                            if random.random() < base_prob * self.fire_intensity[i, j] * dist_factor:
                                new_fire[ni, nj] = self.fire_intensity[i, j] * 0.4
                                if self.fire_ticks[ni, nj] == -1:
                                    self.fire_ticks[ni, nj] = self.current_tick + 1
        
        # Smoke dynamics (simplified but functional)
        self._simulate_smoke_dynamics(new_smoke)
        
        # Update
        self.fire_intensity = new_fire
        self.smoke_intensity = new_smoke
        np.clip(self.fire_intensity, 0.0, 1.0, out=self.fire_intensity)
        np.clip(self.smoke_intensity, 0.0, 1.0, out=self.smoke_intensity)
        
        # Track smoke appearance
        for i in range(self.height):
            for j in range(self.width):
                if self.smoke_ticks[i, j] == -1 and self.smoke_intensity[i, j] > 0.01:
                    self.smoke_ticks[i, j] = self.current_tick
    
    def _simulate_smoke_dynamics(self, new_smoke: np.ndarray):
        """Smoke spread with room equalization"""
        rooms = self._identify_current_rooms()
        
        # Room equalization
        for room_id, cells in rooms.items():
            if not cells:
                continue
            avg = sum(new_smoke[i, j] for i, j in cells) / len(cells)
            for i, j in cells:
                new_smoke[i, j] = new_smoke[i, j] + 0.7 * (avg - new_smoke[i, j])
        
        # Passive dissipation
        for i in range(self.height):
            for j in range(self.width):
                if self.fire_intensity[i, j] <= 0:
                    new_smoke[i, j] = max(0.0, new_smoke[i, j] - 0.01)
    
    def _identify_current_rooms(self) -> Dict[int, Set[Tuple[int, int]]]:
        """Identify rooms considering burned doors"""
        visited = np.zeros((self.height, self.width), dtype=bool)
        rooms = {}
        room_id = 0
        
        for i in range(self.height):
            for j in range(self.width):
                if visited[i, j]:
                    continue
                ct = int(self.floor_plan.grid[i, j])
                if ct in {WALL, WINDOW, EXIT}:
                    continue
                if ct == DOOR and not self.door_burned[i, j]:
                    continue
                
                cells = self._flood_fill_hazard(i, j, visited)
                if cells:
                    rooms[room_id] = cells
                    room_id += 1
        
        return rooms
    
    def _flood_fill_hazard(self, si: int, sj: int, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """Flood fill for current room identification"""
        queue = deque([(si, sj)])
        cells = set()
        
        while queue:
            i, j = queue.popleft()
            if visited[i, j]:
                continue
            
            visited[i, j] = True
            cells.add((i, j))
            
            for di, dj in NEI4:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    continue
                if visited[ni, nj]:
                    continue
                
                nt = int(self.floor_plan.grid[ni, nj])
                if nt in {WALL, WINDOW, EXIT}:
                    continue
                if nt == DOOR and not self.door_burned[ni, nj]:
                    continue
                
                queue.append((ni, nj))
        
        return cells
    
    def _export_visualization(self, stats: Dict, si: int, sj: int):
        """Export D_i visualization"""
        try:
            cs = Config.CELL_SIZE
            surf = pygame.Surface((self.width * cs, self.height * cs))
            surf.fill((255, 255, 255))
            
            # Draw grid
            for i in range(self.height):
                for j in range(self.width):
                    rect = pygame.Rect(j * cs, i * cs, cs - 1, cs - 1)
                    v = int(self.floor_plan.grid[i, j])
                    color = Config.COLORS.get(v, (255, 255, 255))
                    pygame.draw.rect(surf, color, rect)
                    pygame.draw.rect(surf, (150, 150, 150), rect, 1)
            
            # Labels
            font = pygame.font.Font(None, 18)
            for rid, room in self.floor_plan.rooms.items():
                s = stats.get(rid, {"p90": 0})
                center = room.center
                px = int(center[1] * cs + cs / 2)
                py = int(center[0] * cs + cs / 2)
                
                label = f"R{rid} D_i:{s['p90']:.0f}"
                text = font.render(label, True, (255, 0, 0))
                surf.blit(text, text.get_rect(center=(px, py)))
            
            # Fire start
            pygame.draw.circle(surf, (220, 20, 60), (sj * cs + cs // 2, si * cs + cs // 2), 8)
            
            pygame.image.save(surf, "monte_carlo_D_i.jpg")
            print("[MONTE CARLO] Saved monte_carlo_D_i.jpg")
        except Exception as e:
            print(f"[MONTE CARLO] Visualization save failed: {e}")
    
    def get_visual_distance(self, i: int, j: int) -> float:
        """Visual distance based on smoke"""
        smoke = self.smoke_intensity[i, j]
        if smoke < 0.01:
            return 100.0
        C = Config.VISUAL_DISTANCE_CONSTANT
        beta = smoke * 0.26
        return C / beta if beta > 0.001 else 100.0
# ============================================================================
# A* PATHFINDING WITH STRICT FIRE AVOIDANCE
# ============================================================================

class PathFinder:
    """A* pathfinding that STRICTLY avoids fire"""
    
    def __init__(self, floor_plan: FloorPlan, hazard_sim: HazardSimulation):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.height = floor_plan.height
        self.width = floor_plan.width
        
    def find_path(self, start: Tuple[float, float], goal: Tuple[int, int], 
                  allow_fire_fallback: bool = True) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path STRICTLY avoiding fire
        Only allows fire traversal as absolute last resort
        """
        # Convert float start to grid
        start_grid = (int(round(start[0])), int(round(start[1])))
        
        # FIRST ATTEMPT: Completely avoid fire
        path = self._astar(start_grid, goal, allow_fire=False, fire_penalty=0.0)
        
        if path is not None:
            return path
        
        # SECOND ATTEMPT: Avoid fire with very high penalty (only if no alternative)
        if allow_fire_fallback:
            path = self._astar(start_grid, goal, allow_fire=True, fire_penalty=100.0)
            if path:
                print(f"[WARNING] No fire-free path found, using fire path as last resort")
            return path
        
        return None
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int], 
               allow_fire: bool = False, fire_penalty: float = 0.0) -> Optional[List[Tuple[int, int]]]:
        """A* implementation with configurable fire handling"""
        
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            dx = abs(a[1] - b[1])
            dy = abs(a[0] - b[0])
            return (math.sqrt(2) - 1) * min(dx, dy) + max(dx, dy)
        
        open_heap = []
        heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
        
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        
        neighbors = [
            (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
            (0, -1, 1.0), (0, 1, 1.0),
            (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))
        ]
        
        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            if current in closed:
                continue
            closed.add(current)
            
            ci, cj = current
            
            for di, dj, base_cost in neighbors:
                ni, nj = ci + di, cj + dj
                
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    continue
                
                neighbor = (ni, nj)
                cell_type = int(self.floor_plan.grid[ni, nj])
                
                if cell_type in IMPASSABLE_TILES or cell_type == WINDOW:
                    continue
                
                # STRICT FIRE HANDLING
                fire_intensity = self.hazard_sim.fire_intensity[ni, nj]
                if fire_intensity > 0.01:  # Any fire at all
                    if not allow_fire:
                        continue  # COMPLETELY BLOCK fire cells
                    else:
                        # Only when absolutely necessary, with massive penalty
                        pass
                
                # Prevent corner-cutting
                if di != 0 and dj != 0:
                    if (not self.floor_plan.is_walkable(ci + di, cj) or 
                        not self.floor_plan.is_walkable(ci, cj + dj)):
                        continue
                
                # Calculate cost
                step_cost = base_cost
                
                # Apply fire penalty (only when allow_fire=True)
                if fire_intensity > 0.01 and allow_fire:
                    step_cost += fire_penalty * fire_intensity
                
                # Minor smoke penalty (doesn't block, just adds cost)
                smoke_intensity = self.hazard_sim.smoke_intensity[ni, nj]
                step_cost += smoke_intensity * 0.3
                
                tentative_g = g + step_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor))
        
        return None
    
    def calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate path length in meters"""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            i1, j1 = path[i]
            i2, j2 = path[i + 1]
            length += math.hypot(i2 - i1, j2 - j1)
        
        return length
# ============================================================================
# ROOM SEARCH WITH EXACT ROUTE_SIMULATION.PY ALGORITHM
# ============================================================================

def angle_wrap_pi(a):
    """Wrap angle to [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi

def angle_diff(a, b):
    """Calculate angle difference"""
    return angle_wrap_pi(a - b)

def ray_grid_traverse(start_x, start_y, end_x, end_y, grid):
    """Amanatides & Woo voxel traversal (from route_simulation.py)"""
    dx = end_x - start_x
    dy = end_y - start_y
    x = int(math.floor(start_x))
    y = int(math.floor(start_y))
    end_cell_x = int(math.floor(end_x))
    end_cell_y = int(math.floor(end_y))

    if x == end_cell_x and y == end_cell_y:
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            return [(x, y)]
        else:
            return []

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    tDeltaX = abs(1.0 / dx) if dx != 0 else float('inf')
    tDeltaY = abs(1.0 / dy) if dy != 0 else float('inf')

    if dx > 0:
        tMaxX = ((x + 1) - start_x) / dx
    elif dx < 0:
        tMaxX = (x - start_x) / dx
    else:
        tMaxX = float('inf')

    if dy > 0:
        tMaxY = ((y + 1) - start_y) / dy
    elif dy < 0:
        tMaxY = (y - start_y) / dy
    else:
        tMaxY = float('inf')

    cells = []
    max_steps = len(grid) * len(grid[0]) + 5

    for _ in range(max_steps):
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            cells.append((x, y))
        if x == end_cell_x and y == end_cell_y:
            break

        if tMaxX < tMaxY:
            x += step_x
            tMaxX += tDeltaX
        else:
            y += step_y
            tMaxY += tDeltaY

    return cells

def compute_fov_tiles(grid, origin_tile, angle, radius_tiles):
    """Compute visible tiles with 180° FOV (EXACT from route_simulation.py)"""
    ox, oy = origin_tile
    ocx, ocy = ox + 0.5, oy + 0.5

    height = len(grid)
    width = len(grid[0])
    visible = set()

    r = int(radius_tiles)
    min_y = max(0, oy - r)
    max_y = min(height - 1, oy + r)
    min_x = max(0, ox - r)
    max_x = min(width - 1, ox + r)

    half_fov = math.pi / 2
    ANGLE_EPS = 1e-4
    R2 = (radius_tiles + 0.5) ** 2

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if grid[y][x] in IMPASSABLE_TILES:
                continue
            if (x, y) == (ox, oy):
                visible.add((x, y))
                continue

            tcx, tcy = x + 0.5, y + 0.5
            dx = tcx - ocx
            dy = tcy - ocy
            if dx*dx + dy*dy > R2:
                continue

            ang_to_tile = math.atan2(dy, dx)
            if abs(angle_diff(ang_to_tile, angle)) > (half_fov + ANGLE_EPS):
                continue

            # Occlusion test
            blocked = False
            for cx, cy in ray_grid_traverse(ocx, ocy, tcx, tcy, grid):
                if (cx, cy) == (ox, oy):
                    continue
                if (cx, cy) != (x, y) and grid[cy][cx] in IMPASSABLE_TILES:
                    blocked = True
                    break
            if not blocked:
                visible.add((x, y))

    return visible

def nearest_heading_index(angle, heading_count):
    """Map angle to nearest heading index"""
    wrapped = angle % (2 * math.pi)
    idx = int(round(wrapped / (2 * math.pi) * heading_count)) % heading_count
    return idx

def a_star_route(start, end, grid):
    """A* for route planning (from route_simulation.py)"""
    q, visited = [(0, start, [start])], {start}
    neighbors = [
        (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), 
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
    ]
    while q:
        dist, pos, path = heapq.heappop(q)
        if pos == end:
            return dist, path
        for dx, dy, move_len in neighbors:
            nx, ny = pos[0] + dx, pos[1] + dy
            if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                continue
            if grid[ny][nx] in IMPASSABLE_TILES or (nx, ny) in visited:
                continue
            if dx != 0 and dy != 0:
                if (grid[pos[1] + dy][pos[0]] in IMPASSABLE_TILES or 
                    grid[pos[1]][pos[0] + dx] in IMPASSABLE_TILES):
                    continue
            new_dist = dist + move_len
            visited.add((nx, ny))
            priority = new_dist + math.hypot(nx - end[0], ny - end[1])
            heapq.heappush(q, (priority, (nx, ny), path + [(nx, ny)]))
    return float('inf'), []

class RoomSearchCalculator:
    """
    EXACT route_simulation.py algorithm
    NOW: Starts search at closest entry point, not room center
    """
    
    def __init__(self, floor_plan: 'FloorPlan', hazard_sim: 'HazardSimulation'):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.heading_count = Config.HEADING_COUNT
        self.headings = [i * (2 * math.pi / self.heading_count) for i in range(self.heading_count)]
        
    def calculate_search_path(self, room: Room, entry_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate optimal search path using route_simulation.py algorithm
        STARTS AT ENTRY POINT (door or closest room cell to responder)
        Returns complete path including all scan positions
        """
        if not room.cells:
            return [room.center]
        
        # Extract room subgrid
        room_cells_list = list(room.cells)
        min_i = min(c[0] for c in room_cells_list)
        max_i = max(c[0] for c in room_cells_list)
        min_j = min(c[1] for c in room_cells_list)
        max_j = max(c[1] for c in room_cells_list)
        
        # Create local grid for room
        local_height = max_i - min_i + 3
        local_width = max_j - min_j + 3
        local_grid = np.full((local_height, local_width), WALL, dtype=int)
        
        # Fill in room cells
        for ri, rj in room.cells:
            local_i = ri - min_i + 1
            local_j = rj - min_j + 1
            local_grid[local_i, local_j] = int(self.floor_plan.grid[ri, rj])
        
        # Find closest walkable cell to entry point in local coords
        entry_local = (entry_point[0] - min_i + 1, entry_point[1] - min_j + 1)
        
        min_dist = float('inf')
        start_pos = None
        for li in range(local_height):
            for lj in range(local_width):
                if local_grid[li, lj] not in IMPASSABLE_TILES:
                    dist = math.hypot(li - entry_local[0], lj - entry_local[1])
                    if dist < min_dist:
                        min_dist = dist
                        start_pos = (lj, li)  # (x, y) format
        
        if start_pos is None:
            # Fallback to center
            center_local = ((max_i + min_i) // 2 - min_i + 1, 
                           (max_j + min_j) // 2 - min_j + 1)
            start_pos = (center_local[1], center_local[0])
        
        # Run route_simulation.py algorithm
        route_plan = self._find_search_route(local_grid, start_pos)
        
        # Convert back to global coordinates
        global_path = []
        for action in route_plan:
            if action['type'] == 'MOVE':
                for (local_x, local_y) in action['path']:
                    global_i = local_y + min_i - 1
                    global_j = local_x + min_j - 1
                    if (global_i, global_j) in room.cells:
                        global_path.append((global_i, global_j))
            elif action['type'] == 'SCAN':
                pos = action['pos']
                global_i = pos[1] + min_i - 1
                global_j = pos[0] + min_j - 1
                if (global_i, global_j) in room.cells:
                    global_path.append((global_i, global_j))
        
        return global_path if global_path else [room.center]
    
    def _find_search_route(self, grid, start_pos):
        """
        EXACT implementation of route_simulation.py find_smarter_path
        """
        height, width = len(grid), len(grid[0])
        walkable_tiles = {(x, y) for y in range(height) for x in range(width) 
                         if grid[y][x] not in IMPASSABLE_TILES}
        
        uncovered_targets = set(walkable_tiles)
        
        @lru_cache(maxsize=None)
        def cached_visible_from(pos, heading_idx):
            return frozenset(compute_fov_tiles(grid, pos, self.headings[heading_idx], Config.SIGHT_LENGTH))
        
        route_plan = []
        current_pos = start_pos
        current_angle = 0.0
        
        # Initial scan at start position
        initial_idx = nearest_heading_index(current_angle, self.heading_count)
        uncovered_targets -= set(cached_visible_from(current_pos, initial_idx))
        
        iteration = 0
        max_iterations = len(walkable_tiles) * 2
        
        while uncovered_targets and iteration < max_iterations:
            iteration += 1
            
            # Try local scan
            best_local = (0, None)
            for hi, heading_angle in enumerate(self.headings):
                seen = set(cached_visible_from(current_pos, hi))
                newly_seen = seen & uncovered_targets
                if len(newly_seen) > best_local[0]:
                    best_local = (len(newly_seen), hi)
            
            if best_local[0] > 0:
                best_hi = best_local[1]
                best_angle = self.headings[best_hi]
                route_plan.append({'type': 'SCAN', 'pos': current_pos, 'target_angle': best_angle})
                current_angle = best_angle
                uncovered_targets -= set(cached_visible_from(current_pos, best_hi))
                continue
            
            # Find best move
            candidate_moves = []
            for pos in walkable_tiles:
                if pos == current_pos:
                    continue
                dist, path = a_star_route(current_pos, pos, grid)
                if dist == float('inf') or dist <= 0:
                    continue
                
                best_here = None
                for hi, ang in enumerate(self.headings):
                    seen = set(cached_visible_from(pos, hi))
                    newly_seen = seen & uncovered_targets
                    if not newly_seen:
                        continue
                    score = len(newly_seen) / dist
                    cand = {
                        'view': {'pos': pos, 'angle': ang, 'hi': hi}, 
                        'path': path, 
                        'dist': dist, 
                        'score': score,
                        'newly_seen': newly_seen
                    }
                    if best_here is None or score > best_here['score']:
                        best_here = cand
                
                if best_here:
                    candidate_moves.append(best_here)
            
            if not candidate_moves:
                break
            
            # Lookahead selection
            top_candidates = sorted(candidate_moves, key=lambda x: x['score'], reverse=True)[:5]
            best_future_cost = float('inf')
            best_choice = None
            
            for candidate in top_candidates:
                sim_rotation = abs(angle_diff(candidate['view']['angle'], current_angle))
                sim_cost = candidate['dist'] + sim_rotation
                temp_uncovered = uncovered_targets - candidate['newly_seen']
                if temp_uncovered:
                    sim_cost += 1.0 * (len(temp_uncovered) / max(1, len(walkable_tiles)))
                
                if sim_cost < best_future_cost:
                    best_future_cost = sim_cost
                    best_choice = candidate
            
            if best_choice is None:
                break
            
            view, path = best_choice['view'], best_choice['path']
            route_plan.append({'type': 'MOVE', 'path': path})
            route_plan.append({'type': 'SCAN', 'pos': view['pos'], 'target_angle': view['angle']})
            current_pos = view['pos']
            current_angle = view['angle']
            final_scan_seen = set(cached_visible_from(current_pos, view['hi']))
            uncovered_targets -= final_scan_seen
        
        return route_plan
# ============================================================================
# OCCUPANT MANAGEMENT
# ============================================================================
class OccupantManager:
    """Manages all occupants"""
    
    def __init__(self, floor_plan: 'FloorPlan'):
        self.floor_plan = floor_plan
        self.occupants: Dict[int, Occupant] = {}
        self.next_occupant_id = 0
        
    def distribute_occupants(self, occupant_density: float = 0.1, exclude_room: Optional[int] = None):
        """
        Distribute occupants
        ISSUE 6: Exclude fire start room (no one in burning room)
        """
        print("\n[OCCUPANT] Distributing occupants...")
        if exclude_room is not None:
            print(f"[OCCUPANT] Excluding Room {exclude_room} (fire start - D_i=0)")
        
        for room in self.floor_plan.rooms.values():
            # ISSUE 6: Skip fire room OR rooms with D_i = 0
            if room.id == exclude_room:
                room.injured_count = 0
                room.walking_count = 0
                print(f"  Room {room.id}: 0 occupants (FIRE START ROOM)")
                continue
            
            num_occupants = max(1, int(room.area * occupant_density))
            injured_ratio = random.uniform(Config.INJURED_RATIO_MIN, Config.INJURED_RATIO_MAX)
            num_injured = int(num_occupants * injured_ratio)
            num_walking = num_occupants - num_injured
            
            room.injured_count = num_injured
            room.walking_count = num_walking
            
            room_cells = list(room.cells)
            for i in range(num_occupants):
                is_injured = (i < num_injured)
                if room_cells:
                    cell = random.choice(room_cells)
                    position = (float(cell[0]), float(cell[1]))
                else:
                    position = (float(room.center[0]), float(room.center[1]))
                
                occupant = Occupant(
                    id=self.next_occupant_id,
                    room_id=room.id,
                    position=position,
                    is_injured=is_injured
                )
                
                self.occupants[self.next_occupant_id] = occupant
                room.occupants.append(self.next_occupant_id)
                self.next_occupant_id += 1
            
            print(f"  Room {room.id}: {num_occupants} ({num_injured}I + {num_walking}W)")
        
        print(f"[OCCUPANT] Total: {len(self.occupants)}")
    
    def get_waiting_occupants(self, room_id: int) -> Tuple[List[Occupant], List[Occupant]]:
        room = self.floor_plan.rooms.get(room_id)
        if not room:
            return [], []
        
        room_occupants = [self.occupants[oid] for oid in room.occupants if oid in self.occupants]
        injured = [occ for occ in room_occupants if occ.is_injured and occ.state == OccupantState.WAITING]
        walking = [occ for occ in room_occupants if not occ.is_injured and occ.state == OccupantState.WAITING]
        return injured, walking
    
    def assign_occupants_to_responder(self, occupant_ids: List[int], responder_id: int):
        for occ_id in occupant_ids:
            if occ_id in self.occupants:
                occupant = self.occupants[occ_id]
                occupant.assigned_responder = responder_id
                occupant.state = OccupantState.CARRIED if occupant.is_injured else OccupantState.FOLLOWING
    
    def mark_occupants_rescued(self, occupant_ids: List[int]):
        for occ_id in occupant_ids:
            if occ_id in self.occupants:
                self.occupants[occ_id].state = OccupantState.RESCUED
                self.occupants[occ_id].assigned_responder = None
    
    def get_rescued_count(self) -> int:
        return sum(1 for occ in self.occupants.values() if occ.state == OccupantState.RESCUED)
    
    def get_total_count(self) -> int:
        return len(self.occupants)
    
    def all_rescued(self) -> bool:
        return all(occ.state == OccupantState.RESCUED for occ in self.occupants.values())

# ============================================================================
# PRIORITY SYSTEM AND TASK MANAGER (FIXED MULTI-ASSIGNMENT)
# ============================================================================

class PrioritySystem:
    """Dynamic priority using D_i from Monte Carlo"""
    
    def __init__(self, floor_plan: FloorPlan, hazard_sim: HazardSimulation):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.room_priorities: Dict[int, float] = {}
        
    def update_priorities(self, occupant_manager: OccupantManager):
        """Update priorities (Equation 11: Priority_i = P_i / D_i)"""
        for room in self.floor_plan.rooms.values():
            # Count remaining occupants
            P_i = len([oid for oid in room.occupants 
                      if oid in occupant_manager.occupants 
                      and occupant_manager.occupants[oid].state == OccupantState.WAITING])
            
            if P_i == 0:
                room.priority = 0.0
                self.room_priorities[room.id] = 0.0
                continue
            
            # Use D_i from Monte Carlo
            D_i = self.hazard_sim.room_D_i.get(room.id, float('inf'))
            room.time_until_danger = D_i
            
            # Calculate priority (Equation 11)
            if D_i <= 10:
                room.priority = float('inf')  # Emergency
            elif D_i == float('inf'):
                room.priority = 0.001
            else:
                room.priority = P_i / D_i
            
            self.room_priorities[room.id] = room.priority
    
    def get_sorted_rooms(self, available_rooms: Set[int], 
                         responder_position: Tuple[float, float],
                         pathfinder: PathFinder) -> List[int]:
        """
        Get rooms sorted by priority with tiebreakers (Equation 12)
        Only returns rooms that are actually available (not assigned)
        """
        if not available_rooms:
            return []
        
        room_scores = []
        
        for room_id in available_rooms:
            room = self.floor_plan.rooms[room_id]
            priority = self.room_priorities.get(room_id, 0.0)
            
            # Skip if no occupants or already assigned
            if priority == 0.0:
                continue
            if room.status == RoomStatus.ASSIGNED:
                continue
            if room.status == RoomStatus.FULLY_CLEARED:
                continue
            
            # Tiebreakers (Equation 12)
            D_i = room.time_until_danger
            neg_D_i = -D_i if D_i != float('inf') else float('-inf')
            rho_i = len(room.occupants) / room.area if room.area > 0 else 0
            
            # Distance
            path = pathfinder.find_path(responder_position, room.center)
            if not path:
                continue
            
            distance = pathfinder.calculate_path_length(path)
            neg_distance = -distance
            neg_area = -room.area
            
            tiebreaker = (neg_D_i, rho_i, neg_distance, neg_area)
            room_scores.append((priority, tiebreaker, room_id))
        
        # Sort by priority descending, then tiebreakers
        room_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        return [room_id for _, _, room_id in room_scores]


class TaskManager:
    """Central task assignment - PREVENTS MULTI-ASSIGNMENT"""
    
    def __init__(self, floor_plan: FloorPlan, hazard_sim: HazardSimulation,
                 occupant_manager: OccupantManager, pathfinder: PathFinder):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.occupant_manager = occupant_manager
        self.pathfinder = pathfinder
        self.priority_system = PrioritySystem(floor_plan, hazard_sim)
        
        self.unvisited_rooms: Set[int] = set(floor_plan.rooms.keys())
        self.rooms_with_injured: Set[int] = set()
        self.rooms_needing_pickup: Set[int] = set()  # ISSUE 1: Rooms with known wounded locations
        
    def update(self):
        """Update priorities"""
        self.priority_system.update_priorities(self.occupant_manager)
        
        # Track rooms with remaining injured
        self.rooms_with_injured.clear()
        self.rooms_needing_pickup.clear()
        
        for room in self.floor_plan.rooms.values():
            injured, _ = self.occupant_manager.get_waiting_occupants(room.id)
            if injured:
                self.rooms_with_injured.add(room.id)
                # ISSUE 1: If room already searched, it needs direct pickup (no search)
                if room.status in [RoomStatus.PARTIALLY_CLEARED, RoomStatus.SEARCHING]:
                    self.rooms_needing_pickup.add(room.id)
    
    def assign_initial_exits(self, responders: List[Responder]):
        """
        Assign exits based on closest room
        Also pre-assign first room to each responder to prevent overlap
        """
        print("\n[TASK] Assigning optimal starting exits and initial rooms...")
        
        exits = list(self.floor_plan.exits)
        if not exits:
            print("  ERROR: No exits!")
            return
        
        # Get priority-sorted rooms
        sorted_rooms = self.priority_system.get_sorted_rooms(
            self.unvisited_rooms, exits[0], self.pathfinder)
        
        if not sorted_rooms:
            # Fallback: distribute evenly
            for i, responder in enumerate(responders):
                exit_idx = i % len(exits)
                responder.position = (float(exits[exit_idx][0]), float(exits[exit_idx][1]))
                print(f"  R{responder.id} -> exit {exits[exit_idx]} (fallback)")
            return
        
        # PRE-ASSIGN first room to each responder to prevent overlap
        assigned_first_rooms = {}
        
        for i, responder in enumerate(responders):
            # Get the room this responder will target first
            if i < len(sorted_rooms):
                target_room_id = sorted_rooms[i]
                target_room = self.floor_plan.rooms[target_room_id]
                
                # Mark this room as pre-assigned
                assigned_first_rooms[responder.id] = target_room_id
                
                # Find closest exit to this room
                best_exit = exits[0]
                best_distance = float('inf')
                
                for exit_pos in exits:
                    path = self.pathfinder.find_path((float(exit_pos[0]), float(exit_pos[1])), 
                                                    target_room.center)
                    if path:
                        distance = self.pathfinder.calculate_path_length(path)
                        if distance < best_distance:
                            best_distance = distance
                            best_exit = exit_pos
                
                responder.position = (float(best_exit[0]), float(best_exit[1]))
                print(f"  R{responder.id} -> exit {best_exit} (pre-assigned Room {target_room_id})")
            else:
                # More responders than high-priority rooms, distribute to remaining exits
                exit_idx = i % len(exits)
                responder.position = (float(exits[exit_idx][0]), float(exits[exit_idx][1]))
                print(f"  R{responder.id} -> exit {exits[exit_idx]} (overflow)")
        
        # Store pre-assignments for first assignment
        self.initial_room_assignments = assigned_first_rooms    
    def get_next_assignment(self, responder: Responder) -> Optional[int]:
        """
        Get next room assignment
        First assignment uses pre-assigned rooms to prevent overlap
        """
        # Check if this is the first assignment and we have a pre-assignment
        if hasattr(self, 'initial_room_assignments') and responder.id in self.initial_room_assignments:
            if responder.current_trip == 1 and not responder.has_any_occupants():
                # This is the first assignment - use pre-assigned room
                pre_assigned_room = self.initial_room_assignments[responder.id]
                room = self.floor_plan.rooms.get(pre_assigned_room)
                
                # Verify room is still valid and unassigned
                if room and room.assigned_responder is None and pre_assigned_room in self.unvisited_rooms:
                    # Check if room still has occupants
                    injured, walking = self.occupant_manager.get_waiting_occupants(pre_assigned_room)
                    if injured or walking:
                        print(f"[TASK] R{responder.id} using pre-assigned Room {pre_assigned_room}")
                        # Remove from pre-assignments so it won't be used again
                        del self.initial_room_assignments[responder.id]
                        return pre_assigned_room
                
                # Pre-assigned room no longer valid, remove it
                del self.initial_room_assignments[responder.id]
        
        # Normal assignment logic (existing code)
        # Priority 1 - Rooms needing direct pickup
        if self.rooms_needing_pickup:
            available_pickups = {rid for rid in self.rooms_needing_pickup 
                                if self.floor_plan.rooms[rid].assigned_responder is None}
            if available_pickups:
                best_room = None
                best_dist = float('inf')
                for room_id in available_pickups:
                    room = self.floor_plan.rooms[room_id]
                    path = self.pathfinder.find_path(responder.position, room.center)
                    if path:
                        dist = self.pathfinder.calculate_path_length(path)
                        if dist < best_dist:
                            best_dist = dist
                            best_room = room_id
                if best_room is not None:
                    return best_room
        
        # Priority 2: Highest priority unvisited room
        available_rooms = {rid for rid in self.unvisited_rooms 
                          if self.floor_plan.rooms[rid].assigned_responder is None}
        
        sorted_rooms = self.priority_system.get_sorted_rooms(
            available_rooms, responder.position, self.pathfinder)
        
        if sorted_rooms:
            return sorted_rooms[0]
        
        return None
    
    def assign_room_to_responder(self, room_id: int, responder_id: int):
        """Mark room as assigned"""
        room = self.floor_plan.rooms[room_id]
        room.assigned_responder = responder_id
        room.status = RoomStatus.ASSIGNED
    
    def mark_room_visited(self, room_id: int):
        """Mark room as visited"""
        self.unvisited_rooms.discard(room_id)
        room = self.floor_plan.rooms[room_id]
        if room.status == RoomStatus.ASSIGNED:
            room.status = RoomStatus.PARTIALLY_CLEARED
    
    def release_room_assignment(self, room_id: int):
        """Release room assignment"""
        room = self.floor_plan.rooms[room_id]
        room.assigned_responder = None
    
    def is_direct_pickup(self, room_id: int) -> bool:
        """ISSUE 1: Check if room needs direct pickup (no search)"""
        return room_id in self.rooms_needing_pickup
    
    def get_nearest_exit(self, position: Tuple[float, float], exclude_exits: Set[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Find nearest exit with fire-free path
        If exclude_exits provided, skip those exits
        """
        exits = list(self.floor_plan.exits)
        if exclude_exits:
            exits = [e for e in exits if e not in exclude_exits]
        
        if not exits:
            # Fallback to any exit if all excluded
            exits = list(self.floor_plan.exits)
        
        if not exits:
            return (int(round(position[0])), int(round(position[1])))
        
        best_exit = None
        best_distance = float('inf')
        
        # Try each exit, prefer fire-free paths
        for exit_pos in exits:
            path = self.pathfinder.find_path(position, exit_pos, allow_fire_fallback=False)
            if path:
                distance = self.pathfinder.calculate_path_length(path)
                if distance < best_distance:
                    best_distance = distance
                    best_exit = exit_pos
        
        # If no fire-free path found, try with fire allowed
        if best_exit is None:
            for exit_pos in exits:
                path = self.pathfinder.find_path(position, exit_pos, allow_fire_fallback=True)
                if path:
                    distance = self.pathfinder.calculate_path_length(path)
                    if distance < best_distance:
                        best_distance = distance
                        best_exit = exit_pos
        
        return best_exit if best_exit else exits[0]
# ============================================================================
# MAIN SIMULATION ENGINE (UPDATED - FASTER + PROPER TIMING)
# ============================================================================

class EmergencyEvacuationSimulation:
    """Main simulation with all fixes"""
    
    def __init__(self, num_responders: int = None, occupant_density: float = 0.1,
                 fire_start_position: Tuple[int, int] = None):
        
        print("="*80)
        print("EMERGENCY EVACUATION SIMULATION")
        print("="*80)
        
        pygame.init()
        
        # Core components
        self.floor_plan = FloorPlan()
        self.hazard_sim = HazardSimulation(self.floor_plan)
        self.occupant_manager = OccupantManager(self.floor_plan)
        self.pathfinder = PathFinder(self.floor_plan, self.hazard_sim)
        self.search_calculator = RoomSearchCalculator(self.floor_plan, self.hazard_sim)
        
        # Fire start
        if fire_start_position:
            fire_i, fire_j = fire_start_position
        else:
            room = random.choice(list(self.floor_plan.rooms.values()))
            fire_pos = random.choice(list(room.cells))
            fire_i, fire_j = fire_pos
        
        self.fire_start = (fire_i, fire_j)
        self.fire_start_room = self.floor_plan.get_room_for_cell(fire_i, fire_j)
        
        # DON'T distribute occupants yet - wait until after response time
        self.occupant_density = occupant_density
        self.occupants_distributed = False
        
        # Start fire FIRST
        self.hazard_sim.ignite_fire(fire_i, fire_j)
        
        # Monte Carlo
        print("\n" + "="*80)
        print("MONTE CARLO - D_i CALCULATION")
        print("="*80)
        self.hazard_sim.run_monte_carlo_time_to_danger(fire_i, fire_j)
        
        # Calculate responders (using estimate since occupants not distributed yet)
        # We'll recalculate after occupants are distributed
        self.num_responders_target = num_responders
        self.responders: List[Responder] = []
        
        # Task management - will be initialized after occupants distributed
        self.task_manager = None
        
        # Timing
        self.current_time = 0.0
        self.hazard_tick_timer = 0.0
        self.responders_arrived = False
        self.is_running = True
        self.simulation_complete = False
        
        # Statistics
        self.stats = {
            'total_occupants': 0,
            'rescued_occupants': 0,
            'rooms_cleared': 0,
            'total_rooms': len(self.floor_plan.rooms),
            'simulation_time': 0.0,
        }
        
        self._setup_display()
        
        print(f"\n[INIT] Ready")
        print(f"  Fire: ({fire_i}, {fire_j}) in Room {self.fire_start_room}")
        print(f"  Response Time: {Config.RESPONSE_TIME}s")
        print(f"  Occupants will be distributed after {Config.RESPONSE_TIME}s (avoiding burned rooms)")
        print("="*80)    
    def _calculate_required_responders(self) -> int:
        """
        N_total = κ · P_total · (H̄_i · H_T · H_S) × 0.25
        """
        P_total = self.occupant_manager.get_total_count()
        
        if P_total == 0:
            print("\n[RESPONDER CALCULATION] No occupants to rescue!")
            return 1
        
        # Calculate average complexity H̄_i
        if self.floor_plan.rooms:
            H_i_avg = sum(room.complexity_factor for room in self.floor_plan.rooms.values()) / len(self.floor_plan.rooms)
        else:
            H_i_avg = 1.0
        
        # Hazard type factor H_T (fire)
        T_norm = 0.5  # Normalized type (0-1)
        H_T = 1 + Config.ALPHA_H * T_norm
        
        # Hazard severity factor H_S
        S_norm = 0.75  # Normalized severity
        H_S = 1 + Config.BETA_H * S_norm
        
        # Total required responders (Equation 8) × 0.25 multiplier
        N_base = Config.KAPPA * P_total * (H_i_avg * H_T * H_S)
        N_total = N_base * 0.25  # Apply 0.25 multiplier
        
        # Round up and ensure minimum of 1
        N_required = max(1, int(math.ceil(N_total)))
        
        print(f"\n[RESPONDER CALCULATION]")
        print(f"  P_total (Total Occupants): {P_total}")
        print(f"  H̄_i (Avg Complexity): {H_i_avg:.3f}")
        print(f"  H_T (Type Factor): {H_T:.3f} (T_norm={T_norm})")
        print(f"  H_S (Severity Factor): {H_S:.3f} (S_norm={S_norm})")
        print(f"  κ (Base Ratio): {Config.KAPPA}")
        print(f"  Formula: N = κ × P × (H̄ × H_T × H_S) × 0.25")
        print(f"  N_base = {Config.KAPPA} × {P_total} × ({H_i_avg:.3f} × {H_T:.3f} × {H_S:.3f}) = {N_base:.3f}")
        print(f"  N_total = {N_base:.3f} × 0.25 = {N_total:.3f}")
        print(f"  N_required = ceil({N_total:.3f}) = {N_required} responders")
        
        return N_required
    def _distribute_occupants_after_fire_spread(self):
        """
        Distribute occupants AFTER fire has spread for response time
        Exclude all rooms with ANY fire
        """
        print("\n[OCCUPANT] Distributing occupants (excluding burned rooms)...")
        
        # Find all rooms with fire
        rooms_with_fire = set()
        for i in range(self.floor_plan.height):
            for j in range(self.floor_plan.width):
                if self.hazard_sim.fire_intensity[i, j] > 0.01:
                    room_id = self.floor_plan.get_room_for_cell(i, j)
                    if room_id is not None:
                        rooms_with_fire.add(room_id)
        
        print(f"[OCCUPANT] Excluding {len(rooms_with_fire)} rooms with fire: {sorted(rooms_with_fire)}")
        
        # Distribute occupants to safe rooms only
        for room in self.floor_plan.rooms.values():
            if room.id in rooms_with_fire:
                room.injured_count = 0
                room.walking_count = 0
                print(f"  Room {room.id}: 0 occupants (HAS FIRE)")
                continue
            
            num_occupants = max(1, int(room.area * self.occupant_density))
            injured_ratio = random.uniform(Config.INJURED_RATIO_MIN, Config.INJURED_RATIO_MAX)
            num_injured = int(num_occupants * injured_ratio)
            num_walking = num_occupants - num_injured
            
            room.injured_count = num_injured
            room.walking_count = num_walking
            
            room_cells = list(room.cells)
            for i in range(num_occupants):
                is_injured = (i < num_injured)
                if room_cells:
                    cell = random.choice(room_cells)
                    position = (float(cell[0]), float(cell[1]))
                else:
                    position = (float(room.center[0]), float(room.center[1]))
                
                occupant = Occupant(
                    id=self.occupant_manager.next_occupant_id,
                    room_id=room.id,
                    position=position,
                    is_injured=is_injured
                )
                
                self.occupant_manager.occupants[self.occupant_manager.next_occupant_id] = occupant
                room.occupants.append(self.occupant_manager.next_occupant_id)
                self.occupant_manager.next_occupant_id += 1
            
            print(f"  Room {room.id}: {num_occupants} ({num_injured}I + {num_walking}W)")
        
        print(f"[OCCUPANT] Total: {self.occupant_manager.get_total_count()}")
    def _initialize_responders(self, num_responders: int):
        exits = list(self.floor_plan.exits)
        if not exits:
            return
        
        for i in range(num_responders):
            responder = Responder(i, exits[0])
            self.responders.append(responder)
        
        self.task_manager.assign_initial_exits(self.responders)
    
    def _setup_display(self):
        self.cell_size = Config.CELL_SIZE
        self.panel_width = 450
        self.window_width = self.floor_plan.width * self.cell_size + self.panel_width
        self.window_height = self.floor_plan.height * self.cell_size
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Emergency Evacuation Simulation")
        
        self.clock = pygame.time.Clock()
        
        # Dynamic font sizes
        self.font = pygame.font.Font(None, Config.get_font_size(22))
        self.small_font = pygame.font.Font(None, Config.get_font_size(16))
        self.tiny_font = pygame.font.Font(None, Config.get_font_size(12))
        
        print(f"[DISPLAY] Window: {self.window_width}x{self.window_height}, Cell: {self.cell_size}px")
    
    def run(self):
        """Main loop"""
        print("\n[SIMULATION] Starting...")
        print(f"[SIMULATION] Fast-forwarding through {Config.RESPONSE_TIME}s response time...")
        
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
            
            if not self.simulation_complete:
                self.update()
            
            self.render()
            self.clock.tick(Config.TICK_RATE)
        
        self._print_final_statistics()
        pygame.quit()
    
    def update(self):
        """Update with response delay and FAST-FORWARD"""
        dt = (1.0 / Config.TICK_RATE) * Config.TIME_SCALE
        self.current_time += dt
        
        # FAST-FORWARD through response time
        if not self.responders_arrived:
            if self.current_time < Config.RESPONSE_TIME:
                # Calculate how many 1-second "fast-forward" ticks should occur
                while self.current_time < Config.RESPONSE_TIME:
                    self.hazard_sim.spread_hazard()
                    self.current_time += 1.0  # Simulate fast-forwarding 1 second per tick
                
                self.hazard_tick_timer = 0.0  # Ensure smooth transition to normal time
                return
            else:
                # Response time complete
                self.responders_arrived = True
                print(f"\n[RESPONDERS] Arrived at {self.current_time:.1f}s!")
                print(f"[FIRE] Spread through {self.hazard_sim.current_tick} ticks during response time")
                
                # NOW distribute occupants (avoiding rooms with fire)
                if not self.occupants_distributed:
                    self._distribute_occupants_after_fire_spread()
                    self.occupants_distributed = True
                    
                    # Initialize task manager NOW
                    self.task_manager = TaskManager(
                        self.floor_plan, self.hazard_sim, 
                        self.occupant_manager, self.pathfinder
                    )
                    
                    # Calculate and initialize responders
                    if self.num_responders_target is None:
                        num_responders = self._calculate_required_responders()
                    else:
                        num_responders = self.num_responders_target
                    
                    self._initialize_responders(num_responders)
                    self.task_manager.update()
                    
                    # Update stats
                    self.stats['total_occupants'] = self.occupant_manager.get_total_count()
                    
                    print(f"[INIT] Responders: {num_responders}, Occupants: {self.stats['total_occupants']}")        
        # Normal simulation after responders arrived
        # Update hazard every SECONDS_PER_HAZARD_TICK seconds
        self.hazard_tick_timer += dt
        if self.hazard_tick_timer >= Config.SECONDS_PER_HAZARD_TICK:
            self.hazard_sim.spread_hazard()
            self.hazard_tick_timer -= Config.SECONDS_PER_HAZARD_TICK
        
        # Update task manager (only if initialized)
        if self.task_manager:
            self.task_manager.update()
        
        # Update responders (only after arrival and occupants distributed)
        if self.occupants_distributed:
            for responder in self.responders:
                self._update_responder(responder, dt)
        
        # Statistics
        self.stats['rescued_occupants'] = self.occupant_manager.get_rescued_count()
        self.stats['rooms_cleared'] = sum(1 for room in self.floor_plan.rooms.values() 
                                         if room.status == RoomStatus.FULLY_CLEARED)
        self.stats['simulation_time'] = self.current_time
        
        # Check completion
        if self.occupants_distributed and self.occupant_manager.all_rescued():
            self.simulation_complete = True
            print(f"\n[SUCCESS] All rescued at {self.current_time:.1f}s!")
        elif self.current_time > Config.MAX_SIMULATION_TIME:
            self.simulation_complete = True
            print(f"\n[TIMEOUT] Time limit")
    def _update_responder(self, responder: Responder, dt: float):
        """Update responder - FIXED: proper pathing and no idle locks"""
        
        if responder.state == ResponderState.IDLE:
            next_room = self.task_manager.get_next_assignment(responder)
            
            if next_room is not None:
                self.task_manager.assign_room_to_responder(next_room, responder.id)
                responder.assigned_room = next_room
                room = self.floor_plan.rooms[next_room]
                
                # ISSUE 1: Check if this is a direct pickup (no search needed)
                is_pickup = self.task_manager.is_direct_pickup(next_room)
                
                if is_pickup:
                    # Direct pickup - find specific injured occupant location
                    injured, _ = self.occupant_manager.get_waiting_occupants(room.id)
                    if injured:
                        target_pos = (int(round(injured[0].position[0])), 
                                    int(round(injured[0].position[1])))
                        path = self.pathfinder.find_path(responder.position, target_pos)
                        
                        if path:
                            responder.current_path = path
                            responder.path_index = 0
                            responder.state = ResponderState.MOVING_TO_ROOM
                            print(f"[R{responder.id}] -> Room {next_room} for PICKUP at {target_pos}")
                        else:
                            self.task_manager.release_room_assignment(next_room)
                            responder.assigned_room = None
                    else:
                        self.task_manager.release_room_assignment(next_room)
                        responder.assigned_room = None
                else:
                    # Normal search - find entry point FIRST, then path
                    entry_point = self._find_room_entry_point(room, responder.position)
                    path = self.pathfinder.find_path(responder.position, entry_point)
                    
                    if path:
                        responder.current_path = path
                        responder.path_index = 0
                        responder.state = ResponderState.MOVING_TO_ROOM
                        print(f"[R{responder.id}] -> Room {next_room} for SEARCH via {entry_point}")
                    else:
                        print(f"[R{responder.id}] No path to Room {next_room}")
                        self.task_manager.release_room_assignment(next_room)
                        responder.assigned_room = None
            else:
                # NO ASSIGNMENT AVAILABLE
                # If responder needs wounded but can't find a room, just evacuate what they have
                if responder.has_any_occupants():
                    print(f"[R{responder.id}] No rooms available, evacuating current load")
                    responder.needs_wounded = False
                    responder.state = ResponderState.EVACUATING
                    exit_pos = self.task_manager.get_nearest_exit(responder.position)
                    path = self.pathfinder.find_path(responder.position, exit_pos)
                    
                    if path:
                        responder.current_path = path
                        responder.path_index = 0
                    else:
                        # Can't reach exit, just stay idle
                        pass
        
        elif responder.state == ResponderState.MOVING_TO_ROOM:
            if responder.path_index < len(responder.current_path):
                target = responder.current_path[responder.path_index]
                target_pos = (float(target[0]), float(target[1]))
                
                di = target_pos[0] - responder.position[0]
                dj = target_pos[1] - responder.position[1]
                distance = math.hypot(di, dj)
                
                if distance > 0.01:
                    speed = responder.get_speed()
                    move_dist = speed * dt
                    
                    if move_dist >= distance:
                        responder.position = target_pos
                        responder.path_index += 1
                        responder.total_distance_traveled += distance
                    else:
                        ratio = move_dist / distance
                        responder.position = (responder.position[0] + di * ratio, 
                                            responder.position[1] + dj * ratio)
                        responder.total_distance_traveled += move_dist
                else:
                    responder.path_index += 1
            else:
                # Reached destination
                room = self.floor_plan.rooms[responder.assigned_room]
                
                # ISSUE 1: Check if direct pickup or search
                is_pickup = self.task_manager.is_direct_pickup(responder.assigned_room)
                
                if is_pickup:
                    # Direct pickup - immediately rescue
                    self._rescue_occupants_from_room(responder, room)
                    self._decide_next_action_after_rescue(responder, room)
                else:
                    # Check if already searched
                    if room.has_been_searched:
                        # Room already searched, just pick up
                        print(f"[R{responder.id}] Room {responder.assigned_room} already searched, direct pickup")
                        self._rescue_occupants_from_room(responder, room)
                        self._decide_next_action_after_rescue(responder, room)
                    else:
                        # First time search
                        room.status = RoomStatus.SEARCHING
                        room.has_been_searched = True
                        entry_point = (int(round(responder.position[0])), int(round(responder.position[1])))
                        
                        print(f"[R{responder.id}] Generating search path from {entry_point}...")
                        responder.search_path = self.search_calculator.calculate_search_path(room, entry_point)
                        responder.search_path_index = 0
                        responder.state = ResponderState.SEARCHING_ROOM
                        print(f"[R{responder.id}] Search: {len(responder.search_path)} waypoints")
        
        elif responder.state == ResponderState.SEARCHING_ROOM:
            if responder.search_path_index < len(responder.search_path):
                target = responder.search_path[responder.search_path_index]
                target_pos = (float(target[0]), float(target[1]))
                
                di = target_pos[0] - responder.position[0]
                dj = target_pos[1] - responder.position[1]
                distance = math.hypot(di, dj)
                
                if distance > 0.01:
                    search_speed = Config.V1_UNLOADED * 0.6
                    move_dist = search_speed * dt
                    
                    if move_dist >= distance:
                        responder.position = target_pos
                        responder.search_path_index += 1
                        responder.total_distance_traveled += distance
                    else:
                        ratio = move_dist / distance
                        responder.position = (responder.position[0] + di * ratio, 
                                            responder.position[1] + dj * ratio)
                        responder.total_distance_traveled += move_dist
                else:
                    responder.search_path_index += 1
            else:
                # Search complete
                room = self.floor_plan.rooms[responder.assigned_room]
                self._rescue_occupants_from_room(responder, room)
                self._decide_next_action_after_rescue(responder, room)
        
        elif responder.state == ResponderState.EVACUATING:
            # DYNAMIC PATH RECALCULATION with alternative exit finding
            if responder.current_path and responder.path_index < len(responder.current_path):
                # Check next few cells for fire
                needs_replan = False
                lookahead = min(5, len(responder.current_path) - responder.path_index)
                for i in range(responder.path_index, responder.path_index + lookahead):
                    if i >= len(responder.current_path):
                        break
                    cell = responder.current_path[i]
                    if self.hazard_sim.fire_intensity[cell[0], cell[1]] > 0.05:
                        needs_replan = True
                        break
                
                if needs_replan:
                    # Track failed exit attempts to avoid infinite loops
                    if not hasattr(responder, 'failed_exits'):
                        responder.failed_exits = set()
                    if not hasattr(responder, 'replan_count'):
                        responder.replan_count = 0
                    
                    responder.replan_count += 1
                    
                    # If replanned too many times (5+), try different exit
                    if responder.replan_count >= 5:
                        # Get current target exit
                        if responder.current_path:
                            current_target = responder.current_path[-1]
                            responder.failed_exits.add(current_target)
                            print(f"[R{responder.id}] Current exit blocked, trying alternative...")
                        
                        responder.replan_count = 0
                    
                    # Find new exit (excluding failed ones)
                    exit_pos = self.task_manager.get_nearest_exit(responder.position, responder.failed_exits)
                    
                    # Try fire-free path first
                    new_path = self.pathfinder.find_path(responder.position, exit_pos, allow_fire_fallback=False)
                    
                    if new_path:
                        responder.current_path = new_path
                        responder.path_index = 0
                        responder.replan_count = 0
                        responder.failed_exits.clear()  # Reset on success
                        print(f"[R{responder.id}] New FIRE-FREE evacuation path to exit {exit_pos} (length: {len(new_path)})")
                    else:
                        # No fire-free path, allow fire as last resort
                        new_path = self.pathfinder.find_path(responder.position, exit_pos, allow_fire_fallback=True)
                        if new_path:
                            responder.current_path = new_path
                            responder.path_index = 0
                            print(f"[R{responder.id}] Evacuation path with minimal fire to exit {exit_pos} (length: {len(new_path)})")
                        else:
                            # Path completely blocked, mark this exit as failed
                            responder.failed_exits.add(exit_pos)
                            print(f"[R{responder.id}] Exit {exit_pos} completely blocked!")
                            
                            # Try next available exit
                            if len(responder.failed_exits) < len(self.floor_plan.exits):
                                next_exit = self.task_manager.get_nearest_exit(responder.position, responder.failed_exits)
                                print(f"[R{responder.id}] Trying alternative exit: {next_exit}")
            
            # Normal movement along path
            if responder.path_index < len(responder.current_path):
                target = responder.current_path[responder.path_index]
                target_pos = (float(target[0]), float(target[1]))
                
                di = target_pos[0] - responder.position[0]
                dj = target_pos[1] - responder.position[1]
                distance = math.hypot(di, dj)
                
                if distance > 0.01:
                    speed = responder.get_speed()
                    move_dist = speed * dt
                    
                    if move_dist >= distance:
                        responder.position = target_pos
                        responder.path_index += 1
                        responder.total_distance_traveled += distance
                    else:
                        ratio = move_dist / distance
                        responder.position = (responder.position[0] + di * ratio, 
                                            responder.position[1] + dj * ratio)
                        responder.total_distance_traveled += move_dist
                else:
                    responder.path_index += 1
            else:
                # Delivered
                all_occupants = responder.carried_injured + responder.walking_followers
                self.occupant_manager.mark_occupants_rescued(all_occupants)
                
                responder.total_occupants_rescued += len(all_occupants)
                print(f"[R{responder.id}] Delivered {len(all_occupants)}")
                
                responder.carried_injured.clear()
                responder.walking_followers.clear()
                responder.current_trip += 1
                responder.state = ResponderState.IDLE
                responder.assigned_room = None
                responder.needs_wounded = False
                
                # Reset replan tracking
                if hasattr(responder, 'failed_exits'):
                    responder.failed_exits.clear()
                if hasattr(responder, 'replan_count'):
                    responder.replan_count = 0
    
    def _rescue_occupants_from_room(self, responder: Responder, room: Room):
        """Rescue occupants from room"""
        injured, walking = self.occupant_manager.get_waiting_occupants(room.id)
        
        capacity_remaining = Config.MAX_CARRYING_CAPACITY - len(responder.carried_injured)
        injured_to_take = injured[:capacity_remaining]
        
        for occ in injured_to_take:
            responder.carried_injured.append(occ.id)
            room.injured_count -= 1
            room.occupants.remove(occ.id)
        
        for occ in walking:
            responder.walking_followers.append(occ.id)
            room.walking_count -= 1
            room.occupants.remove(occ.id)
        
        all_rescued = responder.carried_injured + responder.walking_followers
        self.occupant_manager.assign_occupants_to_responder(all_rescued, responder.id)
        
        print(f"[R{responder.id}] Rescued {len(injured_to_take)}I + {len(walking)}W")
        
        self.task_manager.mark_room_visited(room.id)
        responder.total_rooms_cleared += 1
        
        if room.injured_count == 0 and room.walking_count == 0:
            room.status = RoomStatus.FULLY_CLEARED
        else:
            room.status = RoomStatus.PARTIALLY_CLEARED
        
        self.task_manager.release_room_assignment(room.id)
    
    def _decide_next_action_after_rescue(self, responder: Responder, room: Room):
        """
        ISSUE 2: Decide whether to evacuate or continue searching
        Continue if: has walking but no injured (needs to fill capacity)
        """
        if responder.should_continue_searching():
            # Has walking followers but no injured - continue to next room
            responder.needs_wounded = True
            responder.state = ResponderState.IDLE
            responder.assigned_room = None
            print(f"[R{responder.id}] Continuing search (needs wounded)")
        else:
            # Evacuate
            responder.needs_wounded = False
            responder.state = ResponderState.EVACUATING
            exit_pos = self.task_manager.get_nearest_exit(responder.position)
            path = self.pathfinder.find_path(responder.position, exit_pos)
            
            if path:
                responder.current_path = path
                responder.path_index = 0
                print(f"[R{responder.id}] Evacuating")
            else:
                responder.state = ResponderState.IDLE
    
    def _find_room_entry_point(self, room: Room, responder_position: Tuple[float, float]) -> Tuple[int, int]:
        """Find closest entry point to room"""
        candidates = []
        
        # Check adjacent doors
        for door in room.adjacent_doors:
            dist = math.hypot(door[0] - responder_position[0], door[1] - responder_position[1])
            candidates.append((dist, door))
        
        # Room cells near doors
        for cell in room.cells:
            is_near_door = False
            for di, dj in NEI8:
                ni, nj = cell[0] + di, cell[1] + dj
                if (ni, nj) in room.adjacent_doors:
                    is_near_door = True
                    break
            
            if is_near_door:
                dist = math.hypot(cell[0] - responder_position[0], cell[1] - responder_position[1])
                candidates.append((dist, cell))
        
        # Fallback: closest room cell
        if not candidates:
            for cell in room.cells:
                dist = math.hypot(cell[0] - responder_position[0], cell[1] - responder_position[1])
                candidates.append((dist, cell))
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else room.center
    def render(self):
        """Render with fire/smoke text overlays"""
        self.screen.fill((255, 255, 255))
        
        self._draw_floor_plan()
        self._draw_hazards_text()
        self._draw_occupants()
        self._draw_responders()
        self._draw_info_panel()
        
        pygame.display.flip()
    
    def _draw_floor_plan(self):
        """Draw floor plan"""
        for i in range(self.floor_plan.height):
            for j in range(self.floor_plan.width):
                x = j * self.cell_size
                y = i * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size - 1, self.cell_size - 1)
                
                cell_type = int(self.floor_plan.grid[i, j])
                color = Config.COLORS.get(cell_type, (200, 200, 200))
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)
    
    def _draw_hazards_text(self):
        """
        Draw fire and smoke as TEXT overlays
        Top-left corner: RED for fire (F:0.XX)
        Below that: BLACK for smoke (S:0.XX)
        """
        for i in range(self.floor_plan.height):
            for j in range(self.floor_plan.width):
                x = j * self.cell_size
                y = i * self.cell_size
                
                # Fire intensity (RED text)
                fire = self.hazard_sim.fire_intensity[i, j]
                if fire > 0.01:
                    fire_text = f"F:{fire:.2f}"
                    fire_surf = self.tiny_font.render(fire_text, True, (255, 0, 0))
                    self.screen.blit(fire_surf, (x + 2, y + 2))
                
                # Smoke density (BLACK text)
                smoke = self.hazard_sim.smoke_intensity[i, j]
                if smoke > 0.05:
                    smoke_text = f"S:{smoke:.2f}"
                    smoke_surf = self.tiny_font.render(smoke_text, True, (0, 0, 0))
                    self.screen.blit(smoke_surf, (x + 2, y + 12))
    
    def _draw_occupants(self):
        """Draw waiting occupants"""
        for occupant in self.occupant_manager.occupants.values():
            if occupant.state != OccupantState.WAITING:
                continue
            
            i, j = occupant.position
            x = int(j * self.cell_size + self.cell_size // 2)
            y = int(i * self.cell_size + self.cell_size // 2)
            
            if occupant.is_injured:
                # Injured: magenta circle
                pygame.draw.circle(self.screen, (255, 0, 255), (x, y), 5)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 5, 1)
            else:
                # Walking: green circle
                pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 4)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 4, 1)
    
    def _draw_responders(self):
        """Draw responders with color-coded paths"""
        for responder in self.responders:
            # Draw path based on state (different colors)
            if responder.state == ResponderState.MOVING_TO_ROOM:
                # Yellow path to room
                if responder.current_path:
                    self._draw_path(responder.current_path[responder.path_index:], 
                                  responder.position, Config.COLORS['path'])
            
            elif responder.state == ResponderState.SEARCHING_ROOM:
                # Cyan search path
                if responder.search_path:
                    self._draw_path(responder.search_path[responder.search_path_index:], 
                                  responder.position, Config.COLORS['search_path'])
            
            elif responder.state == ResponderState.EVACUATING:
                # Green evacuation path
                if responder.current_path:
                    self._draw_path(responder.current_path[responder.path_index:], 
                                  responder.position, Config.COLORS['evac_path'])
            
            # Draw responder (orange circle)
            x = int(responder.position[1] * self.cell_size + self.cell_size // 2)
            y = int(responder.position[0] * self.cell_size + self.cell_size // 2)
            
            pygame.draw.circle(self.screen, Config.COLORS['responder'], (x, y), 10)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 10, 2)
            
            # ID
            id_text = self.small_font.render(str(responder.id), True, (0, 0, 0))
            self.screen.blit(id_text, id_text.get_rect(center=(x, y)))
            
            # Load indicator
            if responder.carried_injured or responder.walking_followers:
                load_text = f"[{len(responder.carried_injured)}I|{len(responder.walking_followers)}W]"
                load_surf = self.tiny_font.render(load_text, True, (255, 0, 0))
                self.screen.blit(load_surf, (x - 18, y - 18))
    
    def _draw_path(self, path: List[Tuple[int, int]], current_pos: Tuple[float, float], 
                   color: Tuple[int, int, int]):
        """Draw path as thick line"""
        if not path:
            return
        
        points = []
        
        # Current position
        curr_x = int(current_pos[1] * self.cell_size + self.cell_size // 2)
        curr_y = int(current_pos[0] * self.cell_size + self.cell_size // 2)
        points.append((curr_x, curr_y))
        
        # Path waypoints
        for pi, pj in path:
            px = int(pj * self.cell_size + self.cell_size // 2)
            py = int(pi * self.cell_size + self.cell_size // 2)
            points.append((px, py))
        
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, 3)
    
    def _draw_info_panel(self):
        """Draw information panel with LARGER, HIGH CONTRAST responder text"""
        panel_x = self.floor_plan.width * self.cell_size
        panel_y = 10
        
        # Background
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (panel_x, 0, self.panel_width, self.window_height))
        
        # Title
        title = self.font.render("EVACUATION", True, (0, 0, 0))
        self.screen.blit(title, (panel_x + 10, panel_y))
        panel_y += 35
        
        # FAST-FORWARD indicator
        if not self.responders_arrived:
            ff_text = f"FAST-FORWARDING..."
            ff_surf = self.font.render(ff_text, True, (255, 100, 0))
            self.screen.blit(ff_surf, (panel_x + 10, panel_y))
            panel_y += 20
            
            progress = (self.current_time / Config.RESPONSE_TIME) * 100
            progress_text = f"Response time: {progress:.1f}%"
            progress_surf = self.small_font.render(progress_text, True, (200, 80, 0))
            self.screen.blit(progress_surf, (panel_x + 10, panel_y))
            panel_y += 10
        
        # Simulation time
        time_text = f"Time: {self.current_time:.1f}s"
        self.screen.blit(self.small_font.render(time_text, True, (0, 0, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 15
        
        # Hazard tick
        hazard_text = f"Hazard Tick: {self.hazard_sim.current_tick}"
        self.screen.blit(self.tiny_font.render(hazard_text, True, (80, 80, 80)), 
                        (panel_x + 10, panel_y))
        panel_y += 15
        
        # Real time equivalent
        real_time_text = f"Real Time: {self.hazard_sim.current_tick * Config.SECONDS_PER_HAZARD_TICK:.1f}s"
        self.screen.blit(self.tiny_font.render(real_time_text, True, (80, 80, 80)), 
                        (panel_x + 10, panel_y))
        panel_y += 15
  
        # Occupants
        rescued_text = f"Rescued: {self.stats['rescued_occupants']}/{self.stats['total_occupants']}"
        self.screen.blit(self.small_font.render(rescued_text, True, (0, 150, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 15
        
        # Progress bar
        if self.stats['total_occupants'] > 0:
            progress = self.stats['rescued_occupants'] / self.stats['total_occupants']
            bar_width = self.panel_width - 40
            pygame.draw.rect(self.screen, (200, 200, 200), 
                           (panel_x + 10, panel_y, bar_width, 20))
            pygame.draw.rect(self.screen, (0, 200, 0), 
                           (panel_x + 10, panel_y, int(bar_width * progress), 20))
            pygame.draw.rect(self.screen, (0, 0, 0), 
                           (panel_x + 10, panel_y, bar_width, 20), 2)
        
        panel_y += 20
        
        # Rooms
        rooms_text = f"Rooms: {self.stats['rooms_cleared']}/{self.stats['total_rooms']}"
        self.screen.blit(self.small_font.render(rooms_text, True, (0, 0, 150)), 
                        (panel_x + 10, panel_y))
        panel_y += 15
        
        # Responders - LARGER TEXT, HIGH CONTRAST
        responder_title = self.small_font.render("RESPONDERS:", True, (0, 0, 0))
        self.screen.blit(responder_title, (panel_x + 10, panel_y))
        panel_y += 15
        
        for responder in self.responders:
            # State with HIGH CONTRAST color coding - LARGER FONT
            state_colors = {
                ResponderState.IDLE: (60, 60, 60),           # Dark gray
                ResponderState.MOVING_TO_ROOM: (200, 150, 0),  # Gold
                ResponderState.SEARCHING_ROOM: (0, 180, 255),  # Bright cyan
                ResponderState.EVACUATING: (0, 200, 0)         # Bright green
            }
            state_color = state_colors.get(responder.state, (0, 0, 0))
            
            # Use SMALL_FONT instead of tiny_font for responder info
            state_text = f"R{responder.id}: {responder.state.value.upper()}"
            text_surf = self.small_font.render(state_text, True, state_color)
            # Add shadow for even better visibility
            shadow_surf = self.small_font.render(state_text, True, (200, 200, 200))
            self.screen.blit(shadow_surf, (panel_x + 16, panel_y + 1))
            self.screen.blit(text_surf, (panel_x + 15, panel_y))
            panel_y += 20
            
            if responder.assigned_room is not None:
                room = self.floor_plan.rooms[responder.assigned_room]
                di_seconds = room.time_until_danger * Config.SECONDS_PER_HAZARD_TICK
                room_text = f"  Rm{responder.assigned_room} D_i:{di_seconds:.0f}s"
                room_surf = self.small_font.render(room_text, True, (40, 40, 40))
                self.screen.blit(room_surf, (panel_x + 20, panel_y))
                panel_y += 18
            
            # Load - HIGH CONTRAST
            load_text = f"  Load: {len(responder.carried_injured)}I+{len(responder.walking_followers)}W"
            load_surf = self.small_font.render(load_text, True, (0, 0, 0))
            self.screen.blit(load_surf, (panel_x + 20, panel_y))
            panel_y += 18
            
            # Saved count - HIGH CONTRAST
            saved_text = f"  Saved: {responder.total_occupants_rescued}"
            saved_surf = self.small_font.render(saved_text, True, (0, 100, 0))
            self.screen.blit(saved_surf, (panel_x + 20, panel_y))
            panel_y += 22
        
        # Fire/Hazard info
        panel_y += 10
        self.screen.blit(self.small_font.render("HAZARD:", True, (0, 0, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 20
        
        fire_cells = np.sum(self.hazard_sim.fire_intensity > 0.01)
        fire_text = f"Fire cells: {fire_cells}"
        self.screen.blit(self.tiny_font.render(fire_text, True, (200, 0, 0)), 
                        (panel_x + 15, panel_y))
        panel_y += 15
        
        smoke_cells = np.sum(self.hazard_sim.smoke_intensity > 0.1)
        smoke_text = f"Smoke cells: {smoke_cells}"
        self.screen.blit(self.tiny_font.render(smoke_text, True, (80, 80, 80)), 
                        (panel_x + 15, panel_y))
        panel_y += 20
        
        # Path legend
        self.screen.blit(self.tiny_font.render("PATHS:", True, (0, 0, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 15
        
        pygame.draw.line(self.screen, Config.COLORS['path'], 
                        (panel_x + 15, panel_y + 5), (panel_x + 40, panel_y + 5), 3)
        self.screen.blit(self.tiny_font.render("Moving", True, (0, 0, 0)), 
                        (panel_x + 45, panel_y))
        panel_y += 13
        
        pygame.draw.line(self.screen, Config.COLORS['search_path'], 
                        (panel_x + 15, panel_y + 5), (panel_x + 40, panel_y + 5), 3)
        self.screen.blit(self.tiny_font.render("Searching", True, (0, 0, 0)), 
                        (panel_x + 45, panel_y))
        panel_y += 13
        
        pygame.draw.line(self.screen, Config.COLORS['evac_path'], 
                        (panel_x + 15, panel_y + 5), (panel_x + 40, panel_y + 5), 3)
        self.screen.blit(self.tiny_font.render("Evacuating", True, (0, 0, 0)), 
                        (panel_x + 45, panel_y))
        
        if self.simulation_complete:
            panel_y += 30
            self.screen.blit(self.font.render("COMPLETE!", True, (0, 150, 0)), 
                           (panel_x + 10, panel_y))
    
    def _print_final_statistics(self):
        """Print final statistics"""
        print("\n" + "="*80)
        print("FINAL STATISTICS")
        print("="*80)
        print(f"Simulation Time: {self.current_time:.2f}s")
        print(f"Hazard Ticks: {self.hazard_sim.current_tick}")
        print(f"Real Time Equivalent: {self.hazard_sim.current_tick * Config.SECONDS_PER_HAZARD_TICK:.1f}s")
        print(f"\nOccupants:")
        print(f"  Total: {self.stats['total_occupants']}")
        print(f"  Rescued: {self.stats['rescued_occupants']}")
        success_rate = 100 * self.stats['rescued_occupants'] / max(1, self.stats['total_occupants'])
        print(f"  Success: {success_rate:.1f}%")
        print(f"\nRooms: {self.stats['rooms_cleared']}/{self.stats['total_rooms']}")
        print(f"\nResponders:")
        for responder in self.responders:
            print(f"  R{responder.id}: {responder.total_occupants_rescued} rescued, "
                  f"{responder.total_rooms_cleared} rooms, {responder.total_distance_traveled:.1f}m")
        print("="*80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    print("EMERGENCY EVACUATION SIMULATION")
    print("HiMCM Mathematical Model Implementation")
    print("  ESC - Exit simulation")
    
    sim = EmergencyEvacuationSimulation(
        num_responders=None,
        occupant_density=0.2,
        fire_start_position=(12, 23)
    )
    
    sim.run()

if __name__ == "__main__":
    main()
