"""
TWO-FLOOR EMERGENCY EVACUATION SIMULATION
Multi-story building with inter-floor fire/smoke propagation
"""

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
    # Grid and Display - TWO FLOORS
    FLOOR_WIDTH = 42
    FLOOR_HEIGHT = 15
    NUM_FLOORS = 2
    
    # Display settings - REVERTED TO FIT SCREEN
    TARGET_WINDOW_HEIGHT = 860  # Smaller height for better fit
    TARGET_WINDOW_WIDTH = 1200  # Smaller width for better fit
    
    # Recalculate cell size based on dimensions
    CELL_SIZE = min(
        TARGET_WINDOW_HEIGHT // (FLOOR_HEIGHT * 2 + 2),
        (TARGET_WINDOW_WIDTH - 400) // FLOOR_WIDTH
    )
    
    FLOOR_GAP = 2  # Cells between floor displays
    
    # Simulation timing
    TICK_RATE = 60
    SECONDS_PER_HAZARD_TICK = 6.0
    TIME_SCALE = 3.0
    MAX_SIMULATION_TIME = 900
    RESPONSE_TIME = 30.0
    
    # Responder parameters
    V1_UNLOADED = 2.0  # m/s on same floor
    V2_WALKING = 1.1
    V3_INJURED = 0.67
    P_STAIRS = 0.82  # Speed multiplier on stairs
    MAX_CARRYING_CAPACITY = 1
    
    # Vertical movement (NEW)
    STAIR_CLIMB_TIME = 15.0  # seconds to go up/down one floor via stairs
    
    # Hazard thresholds
    DOOR_BURN_THRESHOLD = 0.3
    SMOKE_HAZARD_THRESHOLD = 0.3
    
    # Inter-floor fire/smoke spread (NEW)
    VERTICAL_FIRE_SPREAD_PROB = 0.15  # Probability fire spreads up/down through stairs
    VERTICAL_SMOKE_SPREAD_RATE = 0.4   # Rate smoke spreads vertically
    
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
    
    @staticmethod
    def get_font_size(base_size: int) -> int:
        scale_factor = Config.CELL_SIZE / 30.0
        return max(14, int(base_size * scale_factor))  # Minimum font size increased to 14
    
    # Colors
    COLORS = {
        0: (240, 240, 240),  # Empty
        1: (200, 200, 100),  # Light obstacle
        2: (200, 150, 100),  # Medium obstacle
        3: (150, 100, 50),   # Heavy obstacle
        4: (180, 180, 255),  # Staircase (NEW)
        6: (34, 139, 34),    # Exit
        7: (135, 206, 235),  # Window
        8: (100, 100, 200),  # Door
        9: (50, 50, 70),     # Wall
        'fire': (255, 0, 0),
        'smoke': (100, 100, 100),
        'responder': (255, 165, 0),
        'occupant_walking': (0, 255, 0),
        'occupant_injured': (255, 0, 255),
        'path': (255, 215, 0),
        'search_path': (0, 255, 255),
        'evac_path': (0, 255, 0),
        'stair_path': (255, 100, 255)  # Purple for stair movement
    }

# Cell types
WALL = 9
WINDOW = 7
DOOR = 8
EXIT = 6
STAIRCASE = 4  # NEW
EMPTY = 0
LIGHT_OBSTACLE = 1
MEDIUM_OBSTACLE = 2
HEAVY_OBSTACLE = 3

WALKABLE_TILES = {EMPTY, LIGHT_OBSTACLE, MEDIUM_OBSTACLE, DOOR, EXIT, STAIRCASE}
IMPASSABLE_TILES = {WALL, HEAVY_OBSTACLE}

NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# State enumerations
class ResponderState(Enum):
    IDLE = "idle"
    MOVING_TO_ROOM = "moving"
    USING_STAIRS = "stairs"  # NEW
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

@dataclass(frozen=True)  # Add frozen=True to make it immutable and hashable
class Position3D:
    """3D position with floor number"""
    floor: int  # 0 = Floor 1, 1 = Floor 2
    row: float  # Change to float to support fractional positions
    col: float  # Change to float to support fractional positions
    
    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.floor, int(self.row), int(self.col))
    
    def as_2d(self) -> Tuple[int, int]:
        """Return 2D position on current floor"""
        return (int(self.row), int(self.col))
    
    def __hash__(self):
        """Make hashable for use in dictionaries"""
        return hash((self.floor, int(self.row), int(self.col)))
    
    def __eq__(self, other):
        """Equality comparison for A* closed set"""
        if not isinstance(other, Position3D):
            return False
        return (self.floor == other.floor and 
                int(self.row) == int(other.row) and 
                int(self.col) == int(other.col))

@dataclass
class Occupant:
    id: int
    room_id: int  # Global room ID across all floors
    position: Position3D
    is_injured: bool
    state: OccupantState = OccupantState.WAITING
    assigned_responder: Optional[int] = None

@dataclass
class Room:
    id: int  # Global ID
    floor: int  # Which floor this room is on
    cells: Set[Tuple[int, int]]  # 2D cells on that floor
    center: Tuple[int, int]
    area: float
    occupants: List[int]
    injured_count: int
    walking_count: int
    adjacent_doors: Set[Tuple[int, int]]
    adjacent_stairs: Set[Tuple[int, int]]  # NEW
    status: RoomStatus = RoomStatus.UNVISITED
    assigned_responder: Optional[int] = None
    time_until_danger: float = float('inf')
    has_energy_control: bool = False
    has_hazardous_materials: bool = False
    has_confined_space: bool = False
    complexity_factor: float = 1.0
    priority: float = 0.0
    has_been_searched: bool = False
    
    def get_3d_center(self) -> Position3D:
        return Position3D(self.floor, self.center[0], self.center[1])

class Responder:
    def __init__(self, responder_id: int, start_position: Position3D):
        self.id = responder_id
        self.position = start_position
        self.state = ResponderState.IDLE
        self.carried_injured: List[int] = []
        self.walking_followers: List[int] = []
        self.assigned_room: Optional[int] = None
        self.current_path: List[Position3D] = []  # 3D path
        self.path_index: int = 0
        self.search_path: List[Tuple[int, int]] = []  # 2D search within room
        self.search_path_index: int = 0
        self.current_trip: int = 1
        self.total_distance_traveled: float = 0
        self.total_occupants_rescued: int = 0
        self.total_rooms_cleared: int = 0
        self.needs_wounded: bool = False
        
        # Stair traversal tracking
        self.using_stairs: bool = False
        self.stair_start_time: float = 0.0
        self.stair_target_floor: int = 0
        self.stair_position: Tuple[int, int] = (0, 0)
        
    def get_current_load(self) -> int:
        return len(self.carried_injured)
    
    def has_walking_followers(self) -> bool:
        return len(self.walking_followers) > 0
    
    def has_any_occupants(self) -> bool:
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
# TWO-FLOOR FLOOR PLAN MANAGEMENT
# ============================================================================

class TwoFloorPlan:
    """Manages two-floor building layout with vertical connectivity"""
    
    def __init__(self):
        # Generate both floors
        self.floor_grids = self._generate_both_floors()
        self.num_floors = len(self.floor_grids)
        self.height = self.floor_grids[0].shape[0]
        self.width = self.floor_grids[0].shape[1]
        
        # Per-floor special tiles
        self.exits: Dict[int, Set[Tuple[int, int]]] = {0: set(), 1: set()}
        self.doors: Dict[int, Set[Tuple[int, int]]] = {0: set(), 1: set()}
        self.windows: Dict[int, Set[Tuple[int, int]]] = {0: set(), 1: set()}
        self.staircases: Dict[int, Set[Tuple[int, int]]] = {0: set(), 1: set()}
        
        # Global room dictionary (across all floors)
        self.rooms: Dict[int, Room] = {}
        self.next_room_id = 0
        
        # Staircase connectivity map
        self.stair_connections: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        
        self._identify_special_tiles()
        self._identify_stair_connections()
        self._identify_rooms()
        self._calculate_room_properties()
        
    def _generate_both_floors(self) -> List[np.ndarray]:
        """Generate floor plans for both floors"""
        
        # Floor 1 (Ground floor with exits)
        floor1 = np.array([
            [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,6,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [9,0,3,3,3,3,3,0,9,0,2,0,0,0,0,0,9,2,1,0,0,0,1,2,9,0,0,2,2,2,2,2,2,2,2,2,0,0,9,4,4,9],
            [7,0,0,0,0,0,0,0,9,0,2,2,2,2,2,0,9,3,1,0,0,0,1,3,9,0,0,1,1,1,1,1,1,1,1,1,0,0,9,4,4,9],
            [7,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,9,3,1,0,0,0,1,3,9,0,0,1,1,1,1,1,1,1,1,1,0,0,9,0,0,9],
            [9,0,3,3,3,3,3,0,9,0,1,1,0,1,1,0,9,3,1,0,0,0,1,3,9,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,9],
            [9,9,9,9,9,9,9,9,9,0,0,0,0,0,0,0,9,2,1,0,0,0,1,2,9,9,9,9,9,9,8,8,9,9,9,9,9,9,9,0,0,9],
            [9,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,0,8,0,0,9],
            [6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,6],
            [9,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,8,0,0,9],
            [9,9,9,8,8,9,9,9,9,9,9,9,8,8,9,9,9,1,0,0,0,0,0,1,9,9,9,9,8,8,9,9,9,9,9,9,9,8,8,9,9,9],
            [9,2,1,0,0,3,2,2,9,1,1,3,0,0,1,2,9,1,0,0,0,0,0,1,9,2,2,3,0,0,1,2,1,9,2,2,3,0,0,0,0,9],
            [7,2,2,0,0,3,2,2,9,0,0,0,0,0,0,1,9,0,0,0,0,0,0,0,9,2,2,3,0,0,0,1,0,9,2,2,3,0,0,2,2,7],
            [7,2,2,0,0,0,0,0,9,2,2,3,0,0,2,2,9,3,3,3,4,3,3,3,9,0,0,0,0,0,0,2,2,9,0,0,0,0,0,2,2,7],
            [9,2,1,0,0,3,1,1,9,2,2,3,0,0,2,2,9,3,3,3,4,3,3,3,9,1,1,3,0,0,0,2,2,9,1,1,3,0,0,1,2,9],
            [9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9,7,9,9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9]
        ], dtype=int)
        
        # Floor 2 (Upper floor, no ground exits, has staircases)
        floor2 = np.array([
            [9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9,7,9,9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9],
            [9,2,1,0,0,3,2,2,9,2,2,0,0,0,0,0,9,2,2,3,0,0,2,2,9,2,2,3,0,0,0,2,2,9,3,0,0,3,9,4,4,9],
            [7,2,2,0,0,0,2,2,9,2,2,0,0,3,0,0,9,2,2,0,0,0,2,2,9,2,2,0,0,0,0,2,2,9,3,0,0,3,9,4,4,7],
            [7,2,2,0,0,3,3,3,9,1,0,0,0,3,2,2,9,3,3,0,0,0,0,1,9,3,3,0,0,0,0,1,0,9,3,0,0,3,9,0,0,7],
            [9,3,0,0,0,0,0,0,9,2,1,0,0,3,2,2,9,2,1,0,0,0,0,0,9,0,0,0,0,0,1,2,1,9,3,0,0,3,9,0,0,9],
            [9,9,9,9,8,8,9,9,9,9,9,8,8,9,9,9,9,9,9,9,8,8,9,9,9,9,9,8,8,9,9,9,9,9,9,8,8,9,9,0,0,9],
            [7,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,7],
            [7,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,7],
            [7,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,7],
            [9,9,9,9,8,8,9,9,9,9,9,8,8,9,9,9,9,9,9,0,0,0,9,9,9,9,9,8,8,9,9,9,9,9,9,9,9,8,8,9,9,9],
            [9,1,0,0,0,0,0,0,9,2,1,0,0,3,1,0,2,2,9,0,0,0,9,2,2,2,3,0,0,0,0,1,2,9,2,2,3,0,0,0,0,9],
            [7,2,2,0,0,3,3,3,9,1,0,0,0,3,1,0,2,2,9,0,0,0,9,2,2,2,3,0,0,0,0,0,1,9,2,2,3,0,0,2,2,7],
            [7,2,2,0,0,0,2,2,9,2,2,0,0,3,3,0,3,3,9,3,4,3,9,0,0,0,0,0,0,0,2,2,2,9,0,0,0,0,0,2,2,7],
            [9,0,0,0,0,3,2,2,9,2,2,0,0,0,0,0,1,2,9,3,4,3,9,1,1,1,3,0,0,0,2,2,2,9,1,1,3,0,0,1,2,9],
            [9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9,7,9,9,9,9,9,9,9,9,9,9,7,7,9,9,9,9,9,9,9,9,9]
        ], dtype=int)
        
        return [floor1, floor2]
    
    def _identify_special_tiles(self):
        """Identify exits, doors, windows, and staircases on each floor"""
        for floor_num in range(self.num_floors):
            grid = self.floor_grids[floor_num]
            for i in range(self.height):
                for j in range(self.width):
                    cell_type = int(grid[i, j])
                    if cell_type == EXIT:
                        self.exits[floor_num].add((i, j))
                    elif cell_type == DOOR:
                        self.doors[floor_num].add((i, j))
                    elif cell_type == WINDOW:
                        self.windows[floor_num].add((i, j))
                    elif cell_type == STAIRCASE:
                        self.staircases[floor_num].add((i, j))
    
    def _identify_stair_connections(self):
        """Map staircase connections between floors"""
        # Connect corresponding staircase cells between floors
        for floor in range(self.num_floors - 1):
            for stair_pos in self.staircases[floor]:
                # Check if same position exists on floor above
                if stair_pos in self.staircases[floor + 1]:
                    # Create bidirectional connection
                    pos_lower = (floor, stair_pos[0], stair_pos[1])
                    pos_upper = (floor + 1, stair_pos[0], stair_pos[1])
                    self.stair_connections[pos_lower] = pos_upper
                    self.stair_connections[pos_upper] = pos_lower
        
        print(f"[STAIRS] Found {len(self.stair_connections) // 2} staircase connections")
    
    def _identify_rooms(self):
        """Identify rooms on each floor separately"""
        for floor_num in range(self.num_floors):
            visited = np.zeros((self.height, self.width), dtype=bool)
            grid = self.floor_grids[floor_num]
            
            for i in range(self.height):
                for j in range(self.width):
                    cell_type = int(grid[i, j])
                    if visited[i, j] or cell_type in IMPASSABLE_TILES or cell_type in {WINDOW, EXIT}:
                        continue
                    if cell_type in {DOOR, STAIRCASE}:
                        continue
                    
                    # Flood fill to find room
                    cells = self._flood_fill(floor_num, i, j, visited)
                    if cells:
                        center = self._calculate_center(cells)
                        adjacent_doors = self._find_adjacent_doors(floor_num, cells)
                        adjacent_stairs = self._find_adjacent_stairs(floor_num, cells)
                        
                        room = Room(
                            id=self.next_room_id,
                            floor=floor_num,
                            cells=cells,
                            center=center,
                            area=len(cells),
                            occupants=[],
                            injured_count=0,
                            walking_count=0,
                            adjacent_doors=adjacent_doors,
                            adjacent_stairs=adjacent_stairs
                        )
                        self.rooms[self.next_room_id] = room
                        self.next_room_id += 1
        
        print(f"[ROOMS] Identified {len(self.rooms)} rooms across {self.num_floors} floors")
        for floor_num in range(self.num_floors):
            floor_rooms = [r for r in self.rooms.values() if r.floor == floor_num]
            print(f"  Floor {floor_num + 1}: {len(floor_rooms)} rooms")
    
    def _flood_fill(self, floor: int, start_i: int, start_j: int, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """Flood fill on a single floor"""
        queue = deque([(start_i, start_j)])
        cells = set()
        grid = self.floor_grids[floor]
        
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
                
                neighbor_type = int(grid[ni, nj])
                if neighbor_type in IMPASSABLE_TILES or neighbor_type in {WINDOW, EXIT, DOOR, STAIRCASE}:
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
    
    def _find_adjacent_doors(self, floor: int, cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Find all doors adjacent to room cells"""
        adjacent_doors = set()
        grid = self.floor_grids[floor]
        for i, j in cells:
            for di, dj in NEI4:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    if int(grid[ni, nj]) == DOOR:
                        adjacent_doors.add((ni, nj))
        return adjacent_doors
    
    def _find_adjacent_stairs(self, floor: int, cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Find all staircases adjacent to room cells"""
        adjacent_stairs = set()
        grid = self.floor_grids[floor]
        for i, j in cells:
            for di, dj in NEI4:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    if int(grid[ni, nj]) == STAIRCASE:
                        adjacent_stairs.add((ni, nj))
        return adjacent_stairs
    
    def _calculate_room_properties(self):
        """Calculate room areas and complexity factors"""
        for room in self.rooms.values():
            room.area = float(len(room.cells))
            
            if random.random() < 0.2:
                room.has_energy_control = True
            if random.random() < 0.15:
                room.has_hazardous_materials = True
            if random.random() < 0.1:
                room.has_confined_space = True
            
            F1 = 1 if room.has_energy_control else 0
            F2 = 1 if room.has_hazardous_materials else 0
            F3 = 1 if room.has_confined_space else 0
            
            C1, C2, C3 = 0.6, 0.8, 1.0
            room.complexity_factor = 1.0 + Config.GAMMA_H * (F1*C1 + F2*C2 + F3*C3)
    
    def get_room_for_cell(self, floor: int, i: int, j: int) -> Optional[int]:
        """Get room ID for a given 3D cell"""
        for room_id, room in self.rooms.items():
            if room.floor == floor and (i, j) in room.cells:
                return room_id
        return None
    
    def is_walkable(self, floor: int, i: int, j: int) -> bool:
        """Check if 3D cell is walkable"""
        if not (0 <= i < self.height and 0 <= j < self.width):
            return False
        return int(self.floor_grids[floor][i, j]) in WALKABLE_TILES
    
    def get_all_ground_exits(self) -> List[Position3D]:
        """Get all exits on ground floor (floor 0)"""
        return [Position3D(0, i, j) for i, j in self.exits[0]]
    
    def can_use_stairs(self, pos: Position3D) -> bool:
        """Check if position has staircase access"""
        return (pos.floor, pos.row, pos.col) in self.stair_connections
    
    def get_stair_destination(self, pos: Position3D) -> Optional[Position3D]:
        """Get destination floor via stairs"""
        key = (pos.floor, pos.row, pos.col)
        if key in self.stair_connections:
            dest = self.stair_connections[key]
            return Position3D(dest[0], dest[1], dest[2])
        return None
# ============================================================================
# MULTI-FLOOR HAZARD SIMULATION WITH VERTICAL SPREAD
# ============================================================================

class MultiFloorHazardSimulation:
    """Fire and smoke propagation across two floors"""
    
    def __init__(self, floor_plan: TwoFloorPlan):
        self.floor_plan = floor_plan
        self.num_floors = floor_plan.num_floors
        self.height = floor_plan.height
        self.width = floor_plan.width
        
        # Hazard state arrays per floor
        self.fire_intensity = [np.zeros((self.height, self.width), dtype=float) for _ in range(self.num_floors)]
        self.smoke_intensity = [np.zeros((self.height, self.width), dtype=float) for _ in range(self.num_floors)]
        self.fuel_remaining = [np.zeros((self.height, self.width), dtype=float) for _ in range(self.num_floors)]
        self.fuel_initial = [np.zeros((self.height, self.width), dtype=float) for _ in range(self.num_floors)]
        self.door_burned = [np.zeros((self.height, self.width), dtype=bool) for _ in range(self.num_floors)]
        
        # Tracking
        self.fire_ticks = [np.full((self.height, self.width), -1, dtype=int) for _ in range(self.num_floors)]
        self.smoke_ticks = [np.full((self.height, self.width), -1, dtype=int) for _ in range(self.num_floors)]
        self.current_tick = 0
        
        # Fuel properties
        self.fuel_properties = {
            EMPTY: {"max_intensity": 0.6, "burn_rate": 0.10, "fuel_load": 10.0, "smoke_yield": 0.3},
            LIGHT_OBSTACLE: {"max_intensity": 0.9, "burn_rate": 0.25, "fuel_load": 15.0, "smoke_yield": 0.6},
            MEDIUM_OBSTACLE: {"max_intensity": 1.0, "burn_rate": 0.20, "fuel_load": 30.0, "smoke_yield": 1.0},
            HEAVY_OBSTACLE: {"max_intensity": 1.0, "burn_rate": 0.15, "fuel_load": 70.0, "smoke_yield": 1.8},
            DOOR: {"max_intensity": 0.7, "burn_rate": 0.20, "fuel_load": 30.0, "smoke_yield": 0.8},
            STAIRCASE: {"max_intensity": 0.8, "burn_rate": 0.22, "fuel_load": 25.0, "smoke_yield": 0.7},
            WINDOW: {"max_intensity": 0.0, "burn_rate": 0.0, "fuel_load": 0.0, "smoke_yield": 0.0}
        }
        
        # Fire spread probabilities
        self.fire_spread_base_prob = {
            EMPTY: 0.15,
            LIGHT_OBSTACLE: 0.40,
            MEDIUM_OBSTACLE: 0.65,
            HEAVY_OBSTACLE: 0.80,
            DOOR: 0.25,
            STAIRCASE: 0.30  # Slightly higher for vertical spread
        }
        
        self._initialize_fuel()
        self.room_D_i: Dict[int, float] = {}
        
    def _initialize_fuel(self):
        """Initialize fuel loads on all floors"""
        for floor in range(self.num_floors):
            for i in range(self.height):
                for j in range(self.width):
                    cell_type = int(self.floor_plan.floor_grids[floor][i, j])
                    props = self.fuel_properties.get(cell_type, self.fuel_properties[EMPTY])
                    self.fuel_remaining[floor][i, j] = props["fuel_load"]
                    self.fuel_initial[floor][i, j] = props["fuel_load"]
    
    def ignite_fire(self, floor: int, i: int, j: int, intensity: float = 0.5):
        """Start fire at 3D location"""
        if 0 <= floor < self.num_floors and 0 <= i < self.height and 0 <= j < self.width:
            cell_type = int(self.floor_plan.floor_grids[floor][i, j])
            if cell_type not in {WALL, WINDOW}:
                self.fire_intensity[floor][i, j] = intensity
                self.fire_ticks[floor][i, j] = 0
                print(f"[HAZARD] Fire ignited at Floor {floor + 1} ({i}, {j})")
    
    def spread_hazard(self):
        """Advance hazard by one tick with VERTICAL spread"""
        self.current_tick += 1
        
        new_fire = [arr.copy() for arr in self.fire_intensity]
        new_smoke = [arr.copy() for arr in self.smoke_intensity]
        
        # Door burning (per floor)
        for floor in range(self.num_floors):
            for i, j in self.floor_plan.doors[floor]:
                if self.fire_intensity[floor][i, j] > Config.DOOR_BURN_THRESHOLD:
                    self.door_burned[floor][i, j] = True
        
        # Fire growth and horizontal spread (per floor)
        for floor in range(self.num_floors):
            grid = self.floor_plan.floor_grids[floor]
            for i in range(self.height):
                for j in range(self.width):
                    if self.fire_intensity[floor][i, j] > 0:
                        cell_type = int(grid[i, j])
                        props = self.fuel_properties.get(cell_type, self.fuel_properties[EMPTY])
                        
                        max_int = props["max_intensity"]
                        burn_rate = props["burn_rate"]
                        
                        # Growth
                        if max_int > 0 and self.fuel_remaining[floor][i, j] > 0:
                            growth = burn_rate * self.fire_intensity[floor][i, j] * (1 - self.fire_intensity[floor][i, j] / max_int)
                            new_fire[floor][i, j] = min(max_int, self.fire_intensity[floor][i, j] + growth)
                        else:
                            new_fire[floor][i, j] = max(0.0, self.fire_intensity[floor][i, j] - 0.12)
                        
                        # Fuel consumption
                        fuel_cons = min(burn_rate * self.fire_intensity[floor][i, j] * 0.9, self.fuel_remaining[floor][i, j])
                        self.fuel_remaining[floor][i, j] -= fuel_cons
                        
                        # Smoke production
                        if self.fuel_initial[floor][i, j] > 0:
                            smoke_prod = (fuel_cons / self.fuel_initial[floor][i, j]) * props["smoke_yield"] * 25.0
                            new_smoke[floor][i, j] = min(1.0, new_smoke[floor][i, j] + min(smoke_prod, 0.6))
                        
                        # Horizontal fire spread (same floor)
                        for di, dj in NEI8:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if not (0 <= ni < self.height and 0 <= nj < self.width):
                                continue
                            
                            nt = int(grid[ni, nj])
                            if nt in {WALL, WINDOW}:
                                continue
                            
                            if self.fire_intensity[floor][ni, nj] == 0:
                                base_prob = self.fire_spread_base_prob.get(nt, 0.0)
                                dist_factor = 1.0 / (1.0 + math.hypot(di, dj))
                                
                                if random.random() < base_prob * self.fire_intensity[floor][i, j] * dist_factor:
                                    new_fire[floor][ni, nj] = self.fire_intensity[floor][i, j] * 0.4
                                    if self.fire_ticks[floor][ni, nj] == -1:
                                        self.fire_ticks[floor][ni, nj] = self.current_tick + 1
        
        # VERTICAL fire spread through staircases
        for floor in range(self.num_floors):
            for i, j in self.floor_plan.staircases[floor]:
                if self.fire_intensity[floor][i, j] > 0.1:
                    # Check if staircase connects to another floor
                    pos_3d = (floor, i, j)
                    if pos_3d in self.floor_plan.stair_connections:
                        dest = self.floor_plan.stair_connections[pos_3d]
                        dest_floor, dest_i, dest_j = dest
                        
                        # Spread fire vertically (reduced probability)
                        if self.fire_intensity[dest_floor][dest_i, dest_j] == 0:
                            if random.random() < Config.VERTICAL_FIRE_SPREAD_PROB * self.fire_intensity[floor][i, j]:
                                new_fire[dest_floor][dest_i, dest_j] = self.fire_intensity[floor][i, j] * 0.3
                                if self.fire_ticks[dest_floor][dest_i, dest_j] == -1:
                                    self.fire_ticks[dest_floor][dest_i, dest_j] = self.current_tick + 1
                                print(f"[HAZARD] Fire spread from Floor {floor + 1} to Floor {dest_floor + 1} via stairs")
        
        # Smoke dynamics with VERTICAL spread
        self._simulate_smoke_dynamics(new_smoke)
        
        # Update
        self.fire_intensity = new_fire
        self.smoke_intensity = new_smoke
        for floor in range(self.num_floors):
            np.clip(self.fire_intensity[floor], 0.0, 1.0, out=self.fire_intensity[floor])
            np.clip(self.smoke_intensity[floor], 0.0, 1.0, out=self.smoke_intensity[floor])
        
        # Track smoke appearance
        for floor in range(self.num_floors):
            for i in range(self.height):
                for j in range(self.width):
                    if self.smoke_ticks[floor][i, j] == -1 and self.smoke_intensity[floor][i, j] > 0.01:
                        self.smoke_ticks[floor][i, j] = self.current_tick
    
    def _simulate_smoke_dynamics(self, new_smoke: List[np.ndarray]):
        """Smoke spread with room equalization and VERTICAL movement"""
        # Room equalization (per floor)
        for floor in range(self.num_floors):
            floor_rooms = [r for r in self.floor_plan.rooms.values() if r.floor == floor]
            for room in floor_rooms:
                if not room.cells:
                    continue
                avg = sum(new_smoke[floor][i, j] for i, j in room.cells) / len(room.cells)
                for i, j in room.cells:
                    new_smoke[floor][i, j] = new_smoke[floor][i, j] + 0.7 * (avg - new_smoke[floor][i, j])
        
        # VERTICAL smoke spread through staircases (smoke rises faster than fire)
        for floor in range(self.num_floors):
            for i, j in self.floor_plan.staircases[floor]:
                if new_smoke[floor][i, j] > 0.05:
                    pos_3d = (floor, i, j)
                    if pos_3d in self.floor_plan.stair_connections:
                        dest = self.floor_plan.stair_connections[pos_3d]
                        dest_floor, dest_i, dest_j = dest
                        
                        # Smoke rises preferentially (higher rate going up)
                        if dest_floor > floor:  # Going up
                            spread_rate = Config.VERTICAL_SMOKE_SPREAD_RATE * 1.5
                        else:  # Going down
                            spread_rate = Config.VERTICAL_SMOKE_SPREAD_RATE * 0.5
                        
                        smoke_transfer = new_smoke[floor][i, j] * spread_rate
                        new_smoke[dest_floor][dest_i, dest_j] = min(1.0, new_smoke[dest_floor][dest_i, dest_j] + smoke_transfer)
        
        # Passive dissipation
        for floor in range(self.num_floors):
            for i in range(self.height):
                for j in range(self.width):
                    if self.fire_intensity[floor][i, j] <= 0:
                        new_smoke[floor][i, j] = max(0.0, new_smoke[floor][i, j] - 0.01)
    
    def run_monte_carlo_time_to_danger(self, start_floor: int, start_i: int, start_j: int):
        """Monte Carlo simulation for D_i calculation across floors"""
        runs = Config.MONTE_CARLO_RUNS
        max_ticks = Config.MONTE_CARLO_MAX_TICKS
        
        print(f"\n[MONTE CARLO] Running {runs} simulations for D_i calculation (multi-floor)...")
        
        # Save current state
        saved_state = {
            "fire": [arr.copy() for arr in self.fire_intensity],
            "smoke": [arr.copy() for arr in self.smoke_intensity],
            "fuel": [arr.copy() for arr in self.fuel_remaining],
            "door_burned": [arr.copy() for arr in self.door_burned],
            "tick": self.current_tick,
            "fire_ticks": [arr.copy() for arr in self.fire_ticks],
            "smoke_ticks": [arr.copy() for arr in self.smoke_ticks]
        }
        
        # Get room info (all floors)
        per_room_times = {rid: [] for rid in self.floor_plan.rooms}
        
        # All ground floor exits
        ground_exits = set()
        for i, j in self.floor_plan.exits[0]:
            ground_exits.add((0, i, j))
        
        base_seed = int(time.time() * 1000) & 0xffffffff
        
        for run in range(runs):
            random.seed(base_seed ^ (run * 7919))
            
            # Reset to initial state
            self.fire_intensity = [arr.copy() for arr in saved_state["fire"]]
            self.smoke_intensity = [arr.copy() for arr in saved_state["smoke"]]
            self.fuel_remaining = [arr.copy() for arr in saved_state["fuel"]]
            self.door_burned = [arr.copy() for arr in saved_state["door_burned"]]
            self.current_tick = 0
            self.fire_ticks = [arr.copy() for arr in saved_state["fire_ticks"]]
            self.smoke_ticks = [arr.copy() for arr in saved_state["smoke_ticks"]]
            
            # Ignite fire
            self.ignite_fire(start_floor, start_i, start_j, 0.5)
            
            room_hazard_tick = {rid: None for rid in self.floor_plan.rooms}
            
            # Mark starting room as hazardous immediately
            start_room = self.floor_plan.get_room_for_cell(start_floor, start_i, start_j)
            if start_room is not None:
                room_hazard_tick[start_room] = 0
            
            # Simulate hazard spread
            for t in range(max_ticks):
                self.spread_hazard()
                
                # Check hazard conditions for each room
                for rid, room in self.floor_plan.rooms.items():
                    if room_hazard_tick[rid] is not None:
                        continue
                    
                    floor = room.floor
                    
                    # Condition 1: Direct fire in room
                    has_fire = any(self.fire_intensity[floor][ci, cj] > 0 for ci, cj in room.cells)
                    if has_fire:
                        room_hazard_tick[rid] = t
                        continue
                    
                    # Condition 2: Fire at door
                    door_fire = any(self.fire_intensity[floor][di, dj] > Config.DOOR_BURN_THRESHOLD
                                   for di, dj in room.adjacent_doors)
                    if door_fire:
                        room_hazard_tick[rid] = t
                        continue
                    
                    # Condition 3: Critical smoke density
                    avg_smoke = sum(self.smoke_intensity[floor][ci, cj] for ci, cj in room.cells) / len(room.cells)
                    if avg_smoke >= Config.SMOKE_HAZARD_THRESHOLD:
                        room_hazard_tick[rid] = t
                        continue
                    
                    # Condition 4: No safe path to ground exit (3D BFS)
                    has_path = self._check_safe_path_exists_3d(room, ground_exits)
                    if not has_path:
                        room_hazard_tick[rid] = t
                        continue
                
                # Stop if all rooms hazardous
                if all(room_hazard_tick[rid] is not None for rid in self.floor_plan.rooms):
                    break
            
            # Record times
            for rid in self.floor_plan.rooms:
                tick = room_hazard_tick[rid] if room_hazard_tick[rid] is not None else max_ticks
                per_room_times[rid].append(tick)
        
        # Restore state
        self.fire_intensity = [arr.copy() for arr in saved_state["fire"]]
        self.smoke_intensity = [arr.copy() for arr in saved_state["smoke"]]
        self.fuel_remaining = [arr.copy() for arr in saved_state["fuel"]]
        self.door_burned = [arr.copy() for arr in saved_state["door_burned"]]
        self.current_tick = saved_state["tick"]
        self.fire_ticks = [arr.copy() for arr in saved_state["fire_ticks"]]
        self.smoke_ticks = [arr.copy() for arr in saved_state["smoke_ticks"]]
        
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
            self.room_D_i[rid] = p90
        
        print("\n[MONTE CARLO] Time-until-danger (D_i) results (multi-floor):")
        for floor_num in range(self.num_floors):
            print(f"\n  Floor {floor_num + 1}:")
            floor_rooms = [r for r in self.floor_plan.rooms.values() if r.floor == floor_num]
            for room in floor_rooms:
                s = stats[room.id]
                d_i = self.room_D_i[room.id]
                print(f"    Room {room.id}: D_i={d_i:.1f} (min={s['min']}, p90={s['p90']:.1f}, mean={s['mean']:.1f}, max={s['max']})")
        
        return stats
    
    def _check_safe_path_exists_3d(self, room: Room, ground_exits: Set[Tuple[int, int, int]]) -> bool:
        """3D BFS to check if safe path exists from room to any ground exit"""
        if not ground_exits or not room.cells:
            return False
        
        visited = set()
        queue = deque()
        
        # Start from ground exits (reverse search)
        for exit_3d in ground_exits:
            queue.append(exit_3d)
            visited.add(exit_3d)
        
        while queue:
            floor, ci, cj = queue.popleft()
            
            # Found path to room
            if floor == room.floor and (ci, cj) in room.cells:
                return True
            
            # Check horizontal neighbors on same floor
            for di, dj in NEI4:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    continue
                if (floor, ni, nj) in visited:
                    continue
                
                cell_type = int(self.floor_plan.floor_grids[floor][ni, nj])
                if cell_type in {WALL, WINDOW}:
                    continue
                
                # Cannot pass through fire
                if self.fire_intensity[floor][ni, nj] > 0:
                    continue
                
                visited.add((floor, ni, nj))
                queue.append((floor, ni, nj))
            
            # Check vertical movement through staircases
            pos_3d = (floor, ci, cj)
            if pos_3d in self.floor_plan.stair_connections:
                dest = self.floor_plan.stair_connections[pos_3d]
                if dest not in visited:
                    # Check if staircase is passable (no fire)
                    if self.fire_intensity[floor][ci, cj] <= 0 and self.fire_intensity[dest[0]][dest[1], dest[2]] <= 0:
                        visited.add(dest)
                        queue.append(dest)
        
        return False


# ============================================================================
# 3D A* PATHFINDING WITH STAIRS
# ============================================================================

class MultiFloorPathFinder:
    """3D pathfinding across multiple floors"""
    
    def __init__(self, floor_plan: TwoFloorPlan, hazard_sim: MultiFloorHazardSimulation):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.height = floor_plan.height
        self.width = floor_plan.width
        self.num_floors = floor_plan.num_floors
        
    def find_path_3d(self, start: Position3D, goal: Position3D, 
                     allow_fire_fallback: bool = True) -> Optional[List[Position3D]]:
        """Find 3D path with staircase traversal"""
        
        # First attempt: avoid fire completely
        path = self._astar_3d(start, goal, allow_fire=False, fire_penalty=0.0)
        
        if path is not None:
            return path
        
        # Second attempt: allow fire with penalty
        if allow_fire_fallback:
            path = self._astar_3d(start, goal, allow_fire=True, fire_penalty=100.0)
            # REMOVE THE PRINT STATEMENT HERE - let caller handle warnings
            return path
        
        return None
    
    def _astar_3d(self, start: Position3D, goal: Position3D, 
                  allow_fire: bool = False, fire_penalty: float = 0.0) -> Optional[List[Position3D]]:
        """3D A* implementation"""
        
        def heuristic(a: Position3D, b: Position3D) -> float:
            floor_diff = abs(a.floor - b.floor)
            horizontal_dist = math.hypot(a.row - b.row, a.col - b.col)
            stair_cost = floor_diff * (Config.STAIR_CLIMB_TIME / Config.V1_UNLOADED)
            return horizontal_dist + stair_cost
        
        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(start, goal), counter, 0.0, start))
        counter += 1
        
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        
        neighbors_2d = [
            (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
            (0, -1, 1.0), (0, 1, 1.0),
            (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))
        ]
        
        while open_heap:
            f, _, g, current = heapq.heappop(open_heap)
            
            if current.floor == goal.floor and current.row == goal.row and current.col == goal.col:
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
            
            # Horizontal movement on same floor
            for di, dj, base_cost in neighbors_2d:
                ni, nj = current.row + di, current.col + dj
                
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    continue
                
                neighbor = Position3D(current.floor, ni, nj)
                cell_type = int(self.floor_plan.floor_grids[current.floor][int(ni), int(nj)])
                
                if cell_type in IMPASSABLE_TILES or cell_type == WINDOW:
                    continue
                
                # Fire handling - MODIFIED LOGIC
                fire_intensity = self.hazard_sim.fire_intensity[current.floor][int(ni), int(nj)]
                
                # Only block if fire is VERY intense AND we're not allowing fire
                if not allow_fire and fire_intensity > 0.1:  # Changed from 0.01 to 0.1
                    continue
                
                # When allowing fire, only block EXTREME fire (> 0.8)
                if allow_fire and fire_intensity > 0.8:
                    continue
                
                # Prevent corner-cutting
                if di != 0 and dj != 0:
                    if (not self.floor_plan.is_walkable(current.floor, int(current.row + di), int(current.col)) or 
                        not self.floor_plan.is_walkable(current.floor, int(current.row), int(current.col + dj))):
                        continue
                
                step_cost = base_cost
                
                # Add fire penalty
                if fire_intensity > 0.01:
                    if allow_fire:
                        step_cost += fire_penalty * fire_intensity
                    else:
                        step_cost += 10.0 * fire_intensity  # Small penalty for low fire
                
                smoke_intensity = self.hazard_sim.smoke_intensity[current.floor][int(ni), int(nj)]
                step_cost += smoke_intensity * 0.5
                
                tentative_g = g + step_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, counter, tentative_g, neighbor))
                    counter += 1
            
            # Vertical movement through staircases - MODIFIED
            current_3d = (current.floor, int(current.row), int(current.col))
            if current_3d in self.floor_plan.stair_connections:
                dest = self.floor_plan.stair_connections[current_3d]
                neighbor = Position3D(dest[0], dest[1], dest[2])
                
                fire_at_start = self.hazard_sim.fire_intensity[current.floor][int(current.row), int(current.col)]
                fire_at_dest = self.hazard_sim.fire_intensity[dest[0]][dest[1], dest[2]]
                
                # More permissive stair checking
                if not allow_fire and (fire_at_start > 0.1 or fire_at_dest > 0.1):
                    continue
                
                if allow_fire and (fire_at_start > 0.8 or fire_at_dest > 0.8):
                    continue
                
                stair_cost = Config.STAIR_CLIMB_TIME / Config.V1_UNLOADED
                
                # Add fire penalties for stairs
                if fire_at_start > 0.01 or fire_at_dest > 0.01:
                    if allow_fire:
                        stair_cost += fire_penalty * (fire_at_start + fire_at_dest) / 2
                
                tentative_g = g + stair_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, counter, tentative_g, neighbor))
                    counter += 1
        
        return None    
    def calculate_path_length_3d(self, path: List[Position3D]) -> float:
        """Calculate 3D path length including stair time"""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            
            if p1.floor != p2.floor:
                # Stair traversal (time converted to distance)
                length += Config.STAIR_CLIMB_TIME / Config.V1_UNLOADED
            else:
                # Horizontal movement
                length += math.hypot(p2.row - p1.row, p2.col - p1.col)
        
        return length


# ============================================================================
# ROOM SEARCH (SAME AS BEFORE, BUT 2D WITHIN ROOM)
# ============================================================================

def angle_wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def angle_diff(a, b):
    return angle_wrap_pi(a - b)

def ray_grid_traverse(start_x, start_y, end_x, end_y, grid):
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
    wrapped = angle % (2 * math.pi)
    idx = int(round(wrapped / (2 * math.pi) * heading_count)) % heading_count
    return idx

def a_star_route(start, end, grid):
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
    """Room search on a single floor (2D)"""
    
    def __init__(self, floor_plan: TwoFloorPlan, hazard_sim: MultiFloorHazardSimulation):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.heading_count = Config.HEADING_COUNT
        self.headings = [i * (2 * math.pi / self.heading_count) for i in range(self.heading_count)]
        
    def calculate_search_path(self, room: Room, entry_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate search path for a room (2D on its floor)"""
        if not room.cells:
            return [room.center]
        
        room_cells_list = list(room.cells)
        min_i = min(c[0] for c in room_cells_list)
        max_i = max(c[0] for c in room_cells_list)
        min_j = min(c[1] for c in room_cells_list)
        max_j = max(c[1] for c in room_cells_list)
        
        local_height = max_i - min_i + 3
        local_width = max_j - min_j + 3
        local_grid = np.full((local_height, local_width), WALL, dtype=int)
        
        floor_grid = self.floor_plan.floor_grids[room.floor]
        for ri, rj in room.cells:
            local_i = ri - min_i + 1
            local_j = rj - min_j + 1
            local_grid[local_i, local_j] = int(floor_grid[ri, rj])
        
        entry_local = (entry_point[0] - min_i + 1, entry_point[1] - min_j + 1)
        
        min_dist = float('inf')
        start_pos = None
        for li in range(local_height):
            for lj in range(local_width):
                if local_grid[li, lj] not in IMPASSABLE_TILES:
                    dist = math.hypot(li - entry_local[0], lj - entry_local[1])
                    if dist < min_dist:
                        min_dist = dist
                        start_pos = (lj, li)
        
        if start_pos is None:
            center_local = ((max_i + min_i) // 2 - min_i + 1, 
                           (max_j + min_j) // 2 - min_j + 1)
            start_pos = (center_local[1], center_local[0])
        
        route_plan = self._find_search_route(local_grid, start_pos)
        
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
        
        initial_idx = nearest_heading_index(current_angle, self.heading_count)
        uncovered_targets -= set(cached_visible_from(current_pos, initial_idx))
        
        iteration = 0
        max_iterations = len(walkable_tiles) * 2
        
        while uncovered_targets and iteration < max_iterations:
            iteration += 1
            
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
    """Manages all occupants across both floors"""
    
    def __init__(self, floor_plan: TwoFloorPlan):
        self.floor_plan = floor_plan
        self.occupants: Dict[int, Occupant] = {}
        self.next_occupant_id = 0
        
    def distribute_occupants_after_fire_spread(self, occupant_density: float, hazard_sim: MultiFloorHazardSimulation):
        """
        Distribute occupants AFTER fire has spread
        Exclude all rooms with ANY fire
        """
        print("\n[OCCUPANT] Distributing occupants (excluding burned rooms)...")
        
        # Find all rooms with fire
        rooms_with_fire = set()
        for floor in range(self.floor_plan.num_floors):
            for i in range(self.floor_plan.height):
                for j in range(self.floor_plan.width):
                    if hazard_sim.fire_intensity[floor][i, j] > 0.01:
                        room_id = self.floor_plan.get_room_for_cell(floor, i, j)
                        if room_id is not None:
                            rooms_with_fire.add(room_id)
        
        print(f"[OCCUPANT] Excluding {len(rooms_with_fire)} rooms with fire: {sorted(rooms_with_fire)}")
        
        # Distribute occupants to safe rooms only
        for room in self.floor_plan.rooms.values():
            if room.id in rooms_with_fire:
                room.injured_count = 0
                room.walking_count = 0
                print(f"  Room {room.id} (Floor {room.floor + 1}): 0 occupants (HAS FIRE)")
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
                    position = Position3D(room.floor, cell[0], cell[1])
                else:
                    position = Position3D(room.floor, room.center[0], room.center[1])
                
                occupant = Occupant(
                    id=self.next_occupant_id,
                    room_id=room.id,
                    position=position,
                    is_injured=is_injured
                )
                
                self.occupants[self.next_occupant_id] = occupant
                room.occupants.append(self.next_occupant_id)
                self.next_occupant_id += 1
            
            print(f"  Room {room.id} (Floor {room.floor + 1}): {num_occupants} ({num_injured}I + {num_walking}W)")
        
        total = self.get_total_count()
        floor_counts = {}
        for floor in range(self.floor_plan.num_floors):
            count = sum(1 for occ in self.occupants.values() if occ.position.floor == floor)
            floor_counts[floor] = count
        
        print(f"[OCCUPANT] Total: {total}")
        for floor, count in floor_counts.items():
            print(f"  Floor {floor + 1}: {count} occupants")
    
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
# PRIORITY SYSTEM AND TASK MANAGER
# ============================================================================

class PrioritySystem:
    """Dynamic priority using D_i from Monte Carlo"""
    
    def __init__(self, floor_plan: TwoFloorPlan, hazard_sim: MultiFloorHazardSimulation):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.room_priorities: Dict[int, float] = {}
        
    def update_priorities(self, occupant_manager: OccupantManager):
        """Update priorities (Equation 11: Priority_i = P_i / D_i)"""
        for room in self.floor_plan.rooms.values():
            P_i = len([oid for oid in room.occupants 
                      if oid in occupant_manager.occupants 
                      and occupant_manager.occupants[oid].state == OccupantState.WAITING])
            
            if P_i == 0:
                room.priority = 0.0
                self.room_priorities[room.id] = 0.0
                continue
            
            D_i = self.hazard_sim.room_D_i.get(room.id, float('inf'))
            room.time_until_danger = D_i
            
            if D_i <= 10:
                room.priority = float('inf')
            elif D_i == float('inf'):
                room.priority = 0.001
            else:
                room.priority = P_i / D_i
            
            self.room_priorities[room.id] = room.priority
    
    def get_sorted_rooms(self, available_rooms: Set[int], 
                         responder_position: Position3D,
                         pathfinder: MultiFloorPathFinder) -> List[int]:
        """Get rooms sorted by priority with tiebreakers (3D distance)"""
        if not available_rooms:
            return []
        
        # ADD CACHE: Store last calculation
        cache_key = (frozenset(available_rooms), responder_position.as_tuple())
        if hasattr(self, '_room_sort_cache') and hasattr(self, '_cache_time'):
            if self._cache_key == cache_key and time.time() - self._cache_time < 2.0:
                return self._cached_result
        
        room_scores = []
        
        for room_id in available_rooms:
            room = self.floor_plan.rooms[room_id]
            priority = self.room_priorities.get(room_id, 0.0)
            
            if priority == 0.0:
                continue
            if room.status == RoomStatus.ASSIGNED:
                continue
            if room.status == RoomStatus.FULLY_CLEARED:
                continue
            
            D_i = room.time_until_danger
            neg_D_i = -D_i if D_i != float('inf') else float('-inf')
            rho_i = len(room.occupants) / room.area if room.area > 0 else 0
            
            # 3D distance
            room_3d = room.get_3d_center()
            path = pathfinder.find_path_3d(responder_position, room_3d)
            if not path:
                continue
            
            distance = pathfinder.calculate_path_length_3d(path)
            neg_distance = -distance
            neg_area = -room.area
            
            tiebreaker = (neg_D_i, rho_i, neg_distance, neg_area)
            room_scores.append((priority, tiebreaker, room_id))
        
        room_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        result = [room_id for _, _, room_id in room_scores]
        
        # Store in cache
        self._room_sort_cache = True
        self._cache_key = cache_key
        self._cache_time = time.time()
        self._cached_result = result
        
        return result


class TaskManager:
    """Central task assignment for multi-floor building"""
    
    def __init__(self, floor_plan: TwoFloorPlan, hazard_sim: MultiFloorHazardSimulation,
                 occupant_manager: OccupantManager, pathfinder: MultiFloorPathFinder):
        self.floor_plan = floor_plan
        self.hazard_sim = hazard_sim
        self.occupant_manager = occupant_manager
        self.pathfinder = pathfinder
        self.priority_system = PrioritySystem(floor_plan, hazard_sim)
        
        self.unvisited_rooms: Set[int] = set(floor_plan.rooms.keys())
        self.rooms_with_injured: Set[int] = set()
        self.rooms_needing_pickup: Set[int] = set()
        
    def update(self):
        """Update priorities"""
        self.priority_system.update_priorities(self.occupant_manager)
        
        self.rooms_with_injured.clear()
        self.rooms_needing_pickup.clear()
        
        for room in self.floor_plan.rooms.values():
            injured, _ = self.occupant_manager.get_waiting_occupants(room.id)
            if injured:
                self.rooms_with_injured.add(room.id)
                if room.status in [RoomStatus.PARTIALLY_CLEARED, RoomStatus.SEARCHING]:
                    self.rooms_needing_pickup.add(room.id)
    
    def assign_initial_exits(self, responders: List[Responder]):
        """Assign exits and pre-assign first rooms"""
        print("\n[TASK] Assigning optimal starting exits and initial rooms...")
        
        ground_exits = self.floor_plan.get_all_ground_exits()
        if not ground_exits:
            print("  ERROR: No ground exits!")
            return
        
        sorted_rooms = self.priority_system.get_sorted_rooms(
            self.unvisited_rooms, ground_exits[0], self.pathfinder)
        
        if not sorted_rooms:
            for i, responder in enumerate(responders):
                exit_idx = i % len(ground_exits)
                responder.position = ground_exits[exit_idx]
                print(f"  R{responder.id} -> exit {ground_exits[exit_idx].as_2d()} (fallback)")
            return
        
        assigned_first_rooms = {}
        
        for i, responder in enumerate(responders):
            if i < len(sorted_rooms):
                target_room_id = sorted_rooms[i]
                target_room = self.floor_plan.rooms[target_room_id]
                
                assigned_first_rooms[responder.id] = target_room_id
                
                # Find closest ground exit to this room
                best_exit = ground_exits[0]
                best_distance = float('inf')
                
                room_3d = target_room.get_3d_center()
                for exit_pos in ground_exits:
                    path = self.pathfinder.find_path_3d(exit_pos, room_3d)
                    if path:
                        distance = self.pathfinder.calculate_path_length_3d(path)
                        if distance < best_distance:
                            best_distance = distance
                            best_exit = exit_pos
                
                responder.position = best_exit
                print(f"  R{responder.id} -> exit Floor {best_exit.floor + 1} {best_exit.as_2d()} (pre-assigned Room {target_room_id} on Floor {target_room.floor + 1})")
            else:
                exit_idx = i % len(ground_exits)
                responder.position = ground_exits[exit_idx]
                print(f"  R{responder.id} -> exit {ground_exits[exit_idx].as_2d()} (overflow)")
        
        self.initial_room_assignments = assigned_first_rooms
    
    def get_next_assignment(self, responder: Responder) -> Optional[int]:
        if not hasattr(responder, 'assignment_failures'):
            responder.assignment_failures = 0
        
        # If too many failures, wait longer before trying again
        if responder.assignment_failures >= 10:
            print(f"[TASK] R{responder.id} has {responder.assignment_failures} failed assignments, pausing")
            return None
        """Get next room assignment"""
        if hasattr(self, 'initial_room_assignments') and responder.id in self.initial_room_assignments:
            if responder.current_trip == 1 and not responder.has_any_occupants():
                pre_assigned_room = self.initial_room_assignments[responder.id]
                room = self.floor_plan.rooms.get(pre_assigned_room)
                
                if room and room.assigned_responder is None and pre_assigned_room in self.unvisited_rooms:
                    injured, walking = self.occupant_manager.get_waiting_occupants(pre_assigned_room)
                    if injured or walking:
                        print(f"[TASK] R{responder.id} using pre-assigned Room {pre_assigned_room}")
                        del self.initial_room_assignments[responder.id]
                        return pre_assigned_room
                
                del self.initial_room_assignments[responder.id]
        
        # Priority 1 - Rooms needing direct pickup
        if self.rooms_needing_pickup:
            available_pickups = {rid for rid in self.rooms_needing_pickup 
                                if self.floor_plan.rooms[rid].assigned_responder is None}
            if available_pickups:
                best_room = None
                best_dist = float('inf')
                for room_id in available_pickups:
                    room = self.floor_plan.rooms[room_id]
                    room_3d = room.get_3d_center()
                    path = self.pathfinder.find_path_3d(responder.position, room_3d)
                    if path:
                        dist = self.pathfinder.calculate_path_length_3d(path)
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
        room = self.floor_plan.rooms[room_id]
        room.assigned_responder = responder_id
        room.status = RoomStatus.ASSIGNED
    
    def mark_room_visited(self, room_id: int):
        self.unvisited_rooms.discard(room_id)
        room = self.floor_plan.rooms[room_id]
        if room.status == RoomStatus.ASSIGNED:
            room.status = RoomStatus.PARTIALLY_CLEARED
    
    def release_room_assignment(self, room_id: int):
        room = self.floor_plan.rooms[room_id]
        room.assigned_responder = None
    
    def is_direct_pickup(self, room_id: int) -> bool:
        return room_id in self.rooms_needing_pickup
    
    def get_nearest_ground_exit(self, position: Position3D, exclude_exits: Set[Tuple[int, int]] = None) -> Position3D:
        """Find nearest ground exit with fire-free path"""
        ground_exits = self.floor_plan.get_all_ground_exits()
        
        if exclude_exits:
            ground_exits = [e for e in ground_exits if e.as_2d() not in exclude_exits]
        
        # If all exits excluded, reset and try all again
        if not ground_exits:
            ground_exits = self.floor_plan.get_all_ground_exits()
        
        if not ground_exits:
            return Position3D(0, int(round(position.row)), int(round(position.col)))
        
        best_exit = None
        best_distance = float('inf')
        
        # Try fire-free paths first
        for exit_pos in ground_exits:
            path = self.pathfinder.find_path_3d(position, exit_pos, allow_fire_fallback=False)
            if path:
                distance = self.pathfinder.calculate_path_length_3d(path)
                if distance < best_distance:
                    best_distance = distance
                    best_exit = exit_pos
        
        # If no fire-free path, allow fire
        if best_exit is None:
            for exit_pos in ground_exits:
                path = self.pathfinder.find_path_3d(position, exit_pos, allow_fire_fallback=True)
                if path:
                    distance = self.pathfinder.calculate_path_length_3d(path)
                    if distance < best_distance:
                        best_distance = distance
                        best_exit = exit_pos
        
        # CRITICAL FIX: Instead of returning ground_exits[0], try exits in order of proximity
        if best_exit is None:
            # Calculate Euclidean distance as fallback (no path available)
            for exit_pos in ground_exits:
                dist = math.hypot(exit_pos.row - position.row, exit_pos.col - position.col)
                if dist < best_distance:
                    best_distance = dist
                    best_exit = exit_pos
        
        return best_exit if best_exit else ground_exits[0]

# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================

class TwoFloorEvacuationSimulation:
    """Main simulation for two-floor building"""
    
    def __init__(self, num_responders: int = None, occupant_density: float = 0.2,
                 fire_start_floor: int = 1, fire_start_position: Tuple[int, int] = None):
        
        print("="*80)
        print("TWO-FLOOR EMERGENCY EVACUATION SIMULATION")
        print("="*80)
        
        pygame.init()
        
        # Core components
        self.floor_plan = TwoFloorPlan()
        self.hazard_sim = MultiFloorHazardSimulation(self.floor_plan)
        self.occupant_manager = OccupantManager(self.floor_plan)
        self.pathfinder = MultiFloorPathFinder(self.floor_plan, self.hazard_sim)
        self.search_calculator = RoomSearchCalculator(self.floor_plan, self.hazard_sim)
        self.clock = pygame.time.Clock()

        # Fire start (on second floor by default)
        if fire_start_position:
            fire_i, fire_j = fire_start_position
        else:
            # Random room on specified floor
            floor_rooms = [r for r in self.floor_plan.rooms.values() if r.floor == fire_start_floor]
            if floor_rooms:
                room = random.choice(floor_rooms)
                fire_pos = random.choice(list(room.cells))
                fire_i, fire_j = fire_pos
            else:
                fire_i, fire_j = (7, 20)  # Default
        
        self.fire_start = Position3D(fire_start_floor, fire_i, fire_j)
        self.fire_start_room = self.floor_plan.get_room_for_cell(fire_start_floor, fire_i, fire_j)
        
        # Occupants distributed after response time
        self.occupant_density = occupant_density
        self.occupants_distributed = False
        
        # Start fire
        self.hazard_sim.ignite_fire(fire_start_floor, fire_i, fire_j)
        
        # Monte Carlo
        print("\n" + "="*80)
        print("MONTE CARLO - D_i CALCULATION (MULTI-FLOOR)")
        print("="*80)
        self.hazard_sim.run_monte_carlo_time_to_danger(fire_start_floor, fire_i, fire_j)
        
        self.num_responders_target = num_responders
        self.responders: List[Responder] = []
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
        print(f"  Fire: Floor {self.fire_start.floor + 1} {self.fire_start.as_2d()} in Room {self.fire_start_room}")
        print(f"  Response Time: {Config.RESPONSE_TIME}s")
        print(f"  Occupants will be distributed after {Config.RESPONSE_TIME}s")
        print("="*80)
    
    def _calculate_required_responders(self) -> int:
        """Calculate required responder count"""
        P_total = self.occupant_manager.get_total_count()
        
        if P_total == 0:
            print("\n[RESPONDER CALCULATION] No occupants to rescue!")
            return 1
        
        if self.floor_plan.rooms:
            H_i_avg = sum(room.complexity_factor for room in self.floor_plan.rooms.values()) / len(self.floor_plan.rooms)
        else:
            H_i_avg = 1.0
        
        T_norm = 0.5
        H_T = 1 + Config.ALPHA_H * T_norm
        
        S_norm = 0.75
        H_S = 1 + Config.BETA_H * S_norm
        
        N_base = Config.KAPPA * P_total * (H_i_avg * H_T * H_S)
        N_total = N_base * 0.25
        
        N_required = max(1, int(math.ceil(N_total)))
        
        print(f"\n[RESPONDER CALCULATION]")
        print(f"  P_total: {P_total}")
        print(f"  H̄_i: {H_i_avg:.3f}")
        print(f"  H_T: {H_T:.3f}")
        print(f"  H_S: {H_S:.3f}")
        print(f"  N_total = {N_total:.3f} → {N_required} responders")
        
        return N_required
    
    def _initialize_responders(self, num_responders: int):
        ground_exits = self.floor_plan.get_all_ground_exits()
        if not ground_exits:
            return
        
        for i in range(num_responders):
            responder = Responder(i, ground_exits[0])
            self.responders.append(responder)
        
        self.task_manager.assign_initial_exits(self.responders)
    
    def _setup_display(self):
        self.cell_size = Config.CELL_SIZE
        self.panel_width = 500

        # Display both floors VERTICALLY (one above another)
        floor_display_width = self.floor_plan.width * self.cell_size
        floor_display_height = self.floor_plan.height * self.cell_size
        gap_height = Config.FLOOR_GAP * self.cell_size

        # Window dimensions: floors stacked vertically + info panel on right
        self.window_width = floor_display_width + self.panel_width
        self.window_height = floor_display_height * 2 + gap_height

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Two-Floor Emergency Evacuation Simulation")

        # Updated font sizes for better readability
        self.font = pygame.font.Font(None, Config.get_font_size(26))  # Increased font size
        self.small_font = pygame.font.Font(None, Config.get_font_size(20))
        self.tiny_font = pygame.font.Font(None, Config.get_font_size(16))  # Larger tiny font
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
        """Update with fast-forward and 3D movement"""
        dt = (1.0 / Config.TICK_RATE) * Config.TIME_SCALE
        self.current_time += dt
        
        # Fast-forward through response time
        if not self.responders_arrived:
            if self.current_time < Config.RESPONSE_TIME:
                while self.current_time < Config.RESPONSE_TIME:
                    self.hazard_sim.spread_hazard()
                    self.current_time += 1.0
                
                self.hazard_tick_timer = 0.0
                return
            else:
                self.responders_arrived = True
                print(f"\n[RESPONDERS] Arrived at {self.current_time:.1f}s!")
                print(f"[FIRE] Spread through {self.hazard_sim.current_tick} ticks during response time")
                
                if not self.occupants_distributed:
                    self.occupant_manager.distribute_occupants_after_fire_spread(
                        self.occupant_density, self.hazard_sim)
                    self.occupants_distributed = True
                    
                    self.task_manager = TaskManager(
                        self.floor_plan, self.hazard_sim, 
                        self.occupant_manager, self.pathfinder
                    )
                    
                    if self.num_responders_target is None:
                        num_responders = self._calculate_required_responders()
                    else:
                        num_responders = self.num_responders_target
                    
                    self._initialize_responders(num_responders)
                    self.task_manager.update()
                    
                    self.stats['total_occupants'] = self.occupant_manager.get_total_count()
                    
                    print(f"[INIT] Responders: {num_responders}, Occupants: {self.stats['total_occupants']}")
        
        # Normal simulation
        self.hazard_tick_timer += dt
        if self.hazard_tick_timer >= Config.SECONDS_PER_HAZARD_TICK:
            self.hazard_sim.spread_hazard()
            self.hazard_tick_timer -= Config.SECONDS_PER_HAZARD_TICK
        
        if self.task_manager:
            self.task_manager.update()
        
        if self.occupants_distributed:
            for responder in self.responders:
                self._update_responder(responder, dt)
        
        self.stats['rescued_occupants'] = self.occupant_manager.get_rescued_count()
        self.stats['rooms_cleared'] = sum(1 for room in self.floor_plan.rooms.values() 
                                         if room.status == RoomStatus.FULLY_CLEARED)
        self.stats['simulation_time'] = self.current_time
        
        if self.occupants_distributed and self.occupant_manager.all_rescued():
            self.simulation_complete = True
            print(f"\n[SUCCESS] All rescued at {self.current_time:.1f}s!")
        elif self.current_time > Config.MAX_SIMULATION_TIME:
            self.simulation_complete = True
            print(f"\n[TIMEOUT] Time limit")
    def _update_responder(self, responder: Responder, dt: float):
        if not hasattr(responder, '_debug_counter'):
            responder._debug_counter = 0
        responder._debug_counter += 1
        
        if responder._debug_counter >= 60:
            responder._debug_counter = 0
            print(f"[DEBUG R{responder.id}] State: {responder.state.value}, Floor: {responder.position.floor + 1}, "
                  f"Pos: ({int(responder.position.row)}, {int(responder.position.col)}), "
                  f"Load: {len(responder.carried_injured)}I+{len(responder.walking_followers)}W, "
                  f"Path: {len(responder.current_path) if responder.current_path else 0}")
            """Update responder with 3D movement and stair traversal"""
        
        if responder.state == ResponderState.IDLE:
            if responder.has_any_occupants():
                print(f"[R{responder.id}] Occupants loaded, finding path to nearest exit")
                
                if not hasattr(responder, 'failed_exits'):
                    responder.failed_exits = set()
                
                # Get nearest exit (excluding failed ones)
                exit_pos = self.task_manager.get_nearest_ground_exit(responder.position, responder.failed_exits)
                
                # Try fire-free path first
                path = self.pathfinder.find_path_3d(responder.position, exit_pos, allow_fire_fallback=False)
                if not path:
                    # Try with fire allowed
                    path = self.pathfinder.find_path_3d(responder.position, exit_pos, allow_fire_fallback=True)
                
                if path:
                    responder.current_path = path
                    responder.path_index = 0
                    responder.state = ResponderState.EVACUATING
                    # Clear failed exits on successful path
                    responder.failed_exits.clear()
                    print(f"[R{responder.id}] Evacuating with occupants to Floor {exit_pos.floor + 1}, Exit {exit_pos.as_2d()}")
                else:
                    # ADD THIS EXIT TO FAILED LIST
                    responder.failed_exits.add(exit_pos.as_2d())
                    print(f"[R{responder.id}] Exit {exit_pos.as_2d()} blocked, marked as failed ({len(responder.failed_exits)} failed exits)")
                    
                    # If all exits have been tried, reset and try again
                    all_exits = self.floor_plan.get_all_ground_exits()
                    if len(responder.failed_exits) >= len(all_exits):
                        print(f"[R{responder.id}] All exits blocked! Waiting for fire to clear or occupants will not be rescued.")
                        responder.failed_exits.clear()  # Reset to try again
                return
            
            # Otherwise, search for waiting occupants in accessible rooms
            waiting_rooms = [room for room in self.floor_plan.rooms.values() if room.injured_count > 0 or room.walking_count > 0]
            
            if waiting_rooms:
                # Assign the nearest room with waiting occupants
                assigned_room = None
                shortest_path = None
                
                for room in waiting_rooms:
                    if room.assigned_responder is None:
                        entry_point = self._find_room_entry_point(room, responder.position)
                        entry_3d = Position3D(room.floor, entry_point[0], entry_point[1])
                        
                        path = self.pathfinder.find_path_3d(responder.position, entry_3d)
                        if path and (not shortest_path or len(path) < len(shortest_path)):
                            shortest_path = path
                            assigned_room = room
                
                if assigned_room and shortest_path:
                    self.task_manager.assign_room_to_responder(assigned_room.id, responder.id)
                    responder.assigned_room = assigned_room.id
                    responder.current_path = shortest_path
                    responder.path_index = 0
                    responder.state = ResponderState.MOVING_TO_ROOM
                    print(f"[R{responder.id}] Assigned Room {assigned_room.id} (Floor {assigned_room.floor + 1}) with {assigned_room.injured_count}I + {assigned_room.walking_count}W")
                    return
            
            # Throttle idle state assignment checks (avoid spamming CPU)
            if not hasattr(responder, 'last_assignment_check'):
                responder.last_assignment_check = 0.0
            
            if self.current_time - responder.last_assignment_check < 1.0:
                return
            
            responder.last_assignment_check = self.current_time
            next_room = self.task_manager.get_next_assignment(responder)
            
            if next_room is not None:
                self.task_manager.assign_room_to_responder(next_room, responder.id)
                responder.assigned_room = next_room
                room = self.floor_plan.rooms[next_room]
                
                is_pickup = self.task_manager.is_direct_pickup(next_room)
                
                if is_pickup:
                    injured, _ = self.occupant_manager.get_waiting_occupants(room.id)
                    if injured:
                        target_pos = injured[0].position
                        path = self.pathfinder.find_path_3d(responder.position, target_pos)
                        
                        if path:
                            responder.current_path = path
                            responder.path_index = 0
                            responder.state = ResponderState.MOVING_TO_ROOM
                            print(f"[R{responder.id}] Direct pickup at Room {next_room} (Floor {room.floor + 1}) for injured")
                        return
                else:
                    entry_point = self._find_room_entry_point(room, responder.position)
                    entry_3d = Position3D(room.floor, entry_point[0], entry_point[1])
                    path = self.pathfinder.find_path_3d(responder.position, entry_3d)
                    
                    if path:
                        responder.current_path = path
                        responder.path_index = 0
                        responder.state = ResponderState.MOVING_TO_ROOM
                        print(f"[R{responder.id}] Moving to Room {next_room} (Floor {room.floor + 1})")        
        elif responder.state == ResponderState.MOVING_TO_ROOM:
            if responder.path_index < len(responder.current_path):
                target = responder.current_path[responder.path_index]
                
                # Check if next move is stair traversal
                if responder.position.floor != target.floor:
                    # Start stair traversal
                    responder.using_stairs = True
                    responder.stair_start_time = self.current_time
                    responder.stair_target_floor = target.floor
                    responder.stair_position = (target.row, target.col)
                    responder.state = ResponderState.USING_STAIRS
                    print(f"[R{responder.id}] Using stairs: Floor {responder.position.floor + 1} -> Floor {target.floor + 1}")
                    return
                
                # Normal horizontal movement
                di = target.row - responder.position.row
                dj = target.col - responder.position.col
                distance = math.hypot(di, dj)
                
                if distance > 0.01:
                    speed = responder.get_speed()
                    move_dist = speed * dt
                    
                    if move_dist >= distance:
                        responder.position = Position3D(target.floor, target.row, target.col)
                        responder.path_index += 1
                        responder.total_distance_traveled += distance
                    else:
                        ratio = move_dist / distance
                        new_row = responder.position.row + di * ratio
                        new_col = responder.position.col + dj * ratio
                        responder.position = Position3D(floor=responder.position.floor, row=new_row, col=new_col)
                        responder.total_distance_traveled += move_dist
                else:
                    responder.path_index += 1
            else:
                # Reached destination
                room = self.floor_plan.rooms[responder.assigned_room]
                
                is_pickup = self.task_manager.is_direct_pickup(responder.assigned_room)
                
                if is_pickup:
                    self._rescue_occupants_from_room(responder, room)
                    self._decide_next_action_after_rescue(responder, room)
                else:
                    if room.has_been_searched:
                        print(f"[R{responder.id}] Room {responder.assigned_room} already searched, direct pickup")
                        self._rescue_occupants_from_room(responder, room)
                        self._decide_next_action_after_rescue(responder, room)
                    else:
                        room.status = RoomStatus.SEARCHING
                        room.has_been_searched = True
                        entry_point = (int(round(responder.position.row)), int(round(responder.position.col)))
                        
                        print(f"[R{responder.id}] Generating search path...")
                        responder.search_path = self.search_calculator.calculate_search_path(room, entry_point)
                        responder.search_path_index = 0
                        responder.state = ResponderState.SEARCHING_ROOM
                        print(f"[R{responder.id}] Search: {len(responder.search_path)} waypoints")
        
        elif responder.state == ResponderState.USING_STAIRS:
            # Time-based stair traversal
            elapsed = self.current_time - responder.stair_start_time
            
            if elapsed >= Config.STAIR_CLIMB_TIME:
                # Stair traversal complete
                responder.position = Position3D(
                    responder.stair_target_floor,
                    responder.stair_position[0],
                    responder.stair_position[1]
                )
                responder.using_stairs = False
                responder.path_index += 1
                
                # Add stair time to distance (converted)
                responder.total_distance_traveled += Config.STAIR_CLIMB_TIME / Config.V1_UNLOADED
                
                # Determine correct state to return to
                if len(responder.carried_injured) > 0 or len(responder.walking_followers) > 0:
                    responder.state = ResponderState.EVACUATING  # Has occupants - continue evacuating
                elif responder.current_path and responder.path_index < len(responder.current_path):
                    responder.state = ResponderState.MOVING_TO_ROOM  # Still moving to room
                else:
                    responder.state = ResponderState.IDLE  # Path complete, go idle
                
                print(f"[R{responder.id}] Completed stairs, now on Floor {responder.position.floor + 1}, state: {responder.state.value}")
        
        elif responder.state == ResponderState.SEARCHING_ROOM:
            if responder.search_path_index < len(responder.search_path):
                target_2d = responder.search_path[responder.search_path_index]
                
                di = target_2d[0] - responder.position.row
                dj = target_2d[1] - responder.position.col
                distance = math.hypot(di, dj)
                
                if distance > 0.01:
                    search_speed = Config.V1_UNLOADED * 0.6
                    move_dist = search_speed * dt
                    
                    if move_dist >= distance:
                        responder.position = Position3D(responder.position.floor, target_2d[0], target_2d[1])
                        responder.search_path_index += 1
                        responder.total_distance_traveled += distance
                    else:
                        ratio = move_dist / distance
                        new_row = responder.position.row + di * ratio
                        new_col = responder.position.col + dj * ratio
                        responder.position = Position3D(floor=responder.position.floor, row=new_row, col=new_col)
                        responder.total_distance_traveled += move_dist
                else:
                    responder.search_path_index += 1
            else:
                room = self.floor_plan.rooms[responder.assigned_room]
                self._rescue_occupants_from_room(responder, room)
                self._decide_next_action_after_rescue(responder, room)
        
        elif responder.state == ResponderState.EVACUATING:
            # Dynamic path recalculation - ONLY CHECK PERIODICALLY
            if responder.current_path and responder.path_index < len(responder.current_path):
                # Initialize recalculation tracking
                if not hasattr(responder, 'last_replan_check'):
                    responder.last_replan_check = self.current_time
                
                # Only check for replanning every 2 seconds to avoid spam
                if self.current_time - responder.last_replan_check >= 2.0:
                    responder.last_replan_check = self.current_time
                    
                    needs_replan = False
                    lookahead = min(5, len(responder.current_path) - responder.path_index)
                    for i in range(responder.path_index, min(responder.path_index + lookahead, len(responder.current_path))):
                        cell = responder.current_path[i]
                        if self.hazard_sim.fire_intensity[cell.floor][int(cell.row), int(cell.col)] > 0.05:
                            needs_replan = True
                            break
                    
                    if needs_replan:
                        if not hasattr(responder, 'failed_exits'):
                            responder.failed_exits = set()
                        if not hasattr(responder, 'replan_count'):
                            responder.replan_count = 0
                        
                        responder.replan_count += 1
                        
                        if responder.replan_count >= 5:
                            if responder.current_path:
                                current_target = responder.current_path[-1]
                                responder.failed_exits.add(current_target.as_2d())
                                print(f"[R{responder.id}] Current exit blocked, trying alternative...")
                            
                            responder.replan_count = 0
                        
                        exit_pos = self.task_manager.get_nearest_ground_exit(responder.position, responder.failed_exits)
                        
                        new_path = self.pathfinder.find_path_3d(responder.position, exit_pos, allow_fire_fallback=False)
                        
                        if new_path:
                            responder.current_path = new_path
                            responder.path_index = 0
                            responder.replan_count = 0
                            responder.failed_exits.clear()
                            print(f"[R{responder.id}] New FIRE-FREE evacuation path to Floor {exit_pos.floor + 1} {exit_pos.as_2d()} (length: {len(new_path)})")
                        else:
                            new_path = self.pathfinder.find_path_3d(responder.position, exit_pos, allow_fire_fallback=True)
                            if new_path:
                                responder.current_path = new_path
                                responder.path_index = 0
                                print(f"[R{responder.id}] Evacuation path with minimal fire (length: {len(new_path)})")
                            else:
                                responder.failed_exits.add(exit_pos.as_2d())
                                print(f"[R{responder.id}] Exit {exit_pos.as_2d()} completely blocked!")
            
            # Movement logic (same as MOVING_TO_ROOM)
            if responder.path_index < len(responder.current_path):
                target = responder.current_path[responder.path_index]
                
                # Check for stair traversal
                if responder.position.floor != target.floor:
                    responder.using_stairs = True
                    responder.stair_start_time = self.current_time
                    responder.stair_target_floor = target.floor
                    responder.stair_position = (int(target.row), int(target.col))
                    responder.state = ResponderState.USING_STAIRS
                    print(f"[R{responder.id}] Evacuating via stairs: Floor {responder.position.floor + 1} -> Floor {target.floor + 1}")
                    return
                
                di = target.row - responder.position.row
                dj = target.col - responder.position.col
                distance = math.hypot(di, dj)
                
                if distance > 0.01:
                    speed = responder.get_speed()
                    move_dist = speed * dt
                    
                    if move_dist >= distance:
                        responder.position = Position3D(target.floor, target.row, target.col)
                        responder.path_index += 1
                        responder.total_distance_traveled += distance
                    else:
                        ratio = move_dist / distance
                        new_row = responder.position.row + di * ratio
                        new_col = responder.position.col + dj * ratio
                        responder.position = Position3D(floor=responder.position.floor, row=new_row, col=new_col)
                        responder.total_distance_traveled += move_dist
                else:
                    responder.path_index += 1
            else:
                # Delivered to ground exit - IMPROVED RESET LOGIC
                if responder.has_any_occupants():  # ADD THIS CHECK
                    all_occupants = responder.carried_injured + responder.walking_followers
                    self.occupant_manager.mark_occupants_rescued(all_occupants)
                    
                    responder.total_occupants_rescued += len(all_occupants)
                    print(f"[R{responder.id}] Delivered {len(all_occupants)} to ground exit")
                    
                    responder.carried_injured.clear()
                    responder.walking_followers.clear()
                
                # COMPLETE RESET
                responder.current_trip += 1
                responder.state = ResponderState.IDLE
                responder.assigned_room = None
                responder.needs_wounded = False
                responder.current_path = []  # ADD THIS
                responder.path_index = 0      # ADD THIS
                
                # Clear evacuation tracking
                if hasattr(responder, 'failed_exits'):
                    responder.failed_exits.clear()
                if hasattr(responder, 'replan_count'):
                    responder.replan_count = 0
                if hasattr(responder, 'last_replan_check'):
                    del responder.last_replan_check  # ADD THIS
                
                print(f"[R{responder.id}] Reset to IDLE, ready for next assignment")  # ADD THIS
    
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
        
        print(f"[R{responder.id}] Rescued {len(injured_to_take)}I + {len(walking)}W from Room {room.id} (Floor {room.floor + 1})")
        
        self.task_manager.mark_room_visited(room.id)
        responder.total_rooms_cleared += 1
        
        if room.injured_count == 0 and room.walking_count == 0:
            room.status = RoomStatus.FULLY_CLEARED
        else:
            room.status = RoomStatus.PARTIALLY_CLEARED
        
        self.task_manager.release_room_assignment(room.id)
    
    def _decide_next_action_after_rescue(self, responder: Responder, room: Room):
        """Decide whether to evacuate or continue searching"""
        if responder.should_continue_searching():
            responder.needs_wounded = True
            responder.state = ResponderState.IDLE
            responder.assigned_room = None
            print(f"[R{responder.id}] Continuing search (needs wounded)")
        else:
            responder.needs_wounded = False
            responder.state = ResponderState.EVACUATING
            exit_pos = self.task_manager.get_nearest_ground_exit(responder.position)
            path = self.pathfinder.find_path_3d(responder.position, exit_pos)
            
            if path:
                responder.current_path = path
                responder.path_index = 0
                print(f"[R{responder.id}] Evacuating to Floor {exit_pos.floor + 1} {exit_pos.as_2d()}")
            else:
                responder.state = ResponderState.IDLE
    
    def _find_room_entry_point(self, room: Room, responder_position: Position3D) -> Tuple[int, int]:
        """Find closest entry point to room (2D on room's floor)"""
        candidates = []
        
        # Check adjacent doors
        for door in room.adjacent_doors:
            dist = math.hypot(door[0] - responder_position.row, door[1] - responder_position.col)
            # Add floor penalty if on different floor
            if room.floor != responder_position.floor:
                dist += 100.0  # Large penalty for floor difference
            candidates.append((dist, door))
        
        # Check adjacent stairs if responder is on different floor
        if room.floor != responder_position.floor:
            for stair in room.adjacent_stairs:
                dist = math.hypot(stair[0] - responder_position.row, stair[1] - responder_position.col)
                candidates.append((dist, stair))
        
        # Room cells near doors
        for cell in room.cells:
            is_near_door = False
            for di, dj in NEI8:
                ni, nj = cell[0] + di, cell[1] + dj
                if (ni, nj) in room.adjacent_doors:
                    is_near_door = True
                    break
            
            if is_near_door:
                dist = math.hypot(cell[0] - responder_position.row, cell[1] - responder_position.col)
                if room.floor != responder_position.floor:
                    dist += 100.0
                candidates.append((dist, cell))
        
        # Fallback: closest room cell
        if not candidates:
            for cell in room.cells:
                dist = math.hypot(cell[0] - responder_position.row, cell[1] - responder_position.col)
                if room.floor != responder_position.floor:
                    dist += 100.0
                candidates.append((dist, cell))
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else room.center
    
    def render(self):
        """Render both floors stacked vertically"""
        self.screen.fill((255, 255, 255))
        
        # Calculate floor positions (VERTICAL stacking)
        floor_display_height = self.floor_plan.height * self.cell_size
        gap_height = Config.FLOOR_GAP * self.cell_size
        
        floor1_y_offset = 0
        floor2_y_offset = floor_display_height + gap_height
        
        # Draw both floors
        self._draw_floor(0, 0, floor1_y_offset)
        self._draw_floor(1, 0, floor2_y_offset)
        
        # Draw floor labels
        label1 = self.font.render("FLOOR 1 (GROUND)", True, (0, 0, 0))
        label2 = self.font.render("FLOOR 2", True, (0, 0, 0))
        self.screen.blit(label1, (10, floor1_y_offset + 10))
        self.screen.blit(label2, (10, floor2_y_offset + 10))
        
        # Draw responders and occupants
        self._draw_occupants_all_floors(0, floor1_y_offset, floor2_y_offset)
        self._draw_responders_all_floors(0, floor1_y_offset, floor2_y_offset)
        
        # Draw info panel (on the right side)
        panel_x = self.floor_plan.width * self.cell_size + 10
        self._draw_info_panel(panel_x)
        
        pygame.display.flip()    
    def _draw_floor(self, floor_num: int, x_offset: int, y_offset: int):
        """Draw a single floor at specified position"""
        grid = self.floor_plan.floor_grids[floor_num]
        
        for i in range(self.floor_plan.height):
            for j in range(self.floor_plan.width):
                x = x_offset + j * self.cell_size
                y = y_offset + i * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size - 1, self.cell_size - 1)
                
                cell_type = int(grid[i, j])
                color = Config.COLORS.get(cell_type, (200, 200, 200))
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)
                
                # Draw hazards as minimal text with smaller font
                fire = self.hazard_sim.fire_intensity[floor_num][i, j]
                if fire > 0.01:
                    fire_surf = self.tiny_font.render(f"{fire:.2f}", True, (255, 0, 0))
                    self.screen.blit(fire_surf, (x + 2, y + 2))
                
                smoke = self.hazard_sim.smoke_intensity[floor_num][i, j]
                if smoke > 0.05:
                    smoke_surf = self.tiny_font.render(f"{smoke:.2f}", True, (100, 100, 100))
                    self.screen.blit(smoke_surf, (x + 2, y + 12))
    
    def _draw_occupants_all_floors(self, x_offset: int, floor1_y_offset: int, floor2_y_offset: int):
        """Draw occupants on both floors"""
        y_offsets = [floor1_y_offset, floor2_y_offset]
        
        for occupant in self.occupant_manager.occupants.values():
            if occupant.state != OccupantState.WAITING:
                continue
            
            floor = occupant.position.floor
            y_offset = y_offsets[floor]
            
            x = int(x_offset + occupant.position.col * self.cell_size + self.cell_size // 2)
            y = int(y_offset + occupant.position.row * self.cell_size + self.cell_size // 2)
            
            if occupant.is_injured:
                pygame.draw.circle(self.screen, (255, 0, 255), (x, y), 5)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 5, 1)
            else:
                pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 4)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 4, 1)
    
    def _draw_responders_all_floors(self, x_offset: int, floor1_y_offset: int, floor2_y_offset: int):
        """Draw responders on their current floor"""
        y_offsets = [floor1_y_offset, floor2_y_offset]
        
        for responder in self.responders:
            floor = responder.position.floor
            y_offset = y_offsets[floor]
            
            # Draw path if exists
            if responder.current_path and responder.state in [ResponderState.MOVING_TO_ROOM, ResponderState.EVACUATING]:
                self._draw_path_3d(responder.current_path[responder.path_index:], 
                                  responder.position, x_offset, y_offsets)
            
            # Draw responder
            x = int(x_offset + responder.position.col * self.cell_size + self.cell_size // 2)
            y = int(y_offset + responder.position.row * self.cell_size + self.cell_size // 2)
            
            # Different color if using stairs
            if responder.state == ResponderState.USING_STAIRS:
                color = Config.COLORS['stair_path']
            else:
                color = Config.COLORS['responder']
            
            pygame.draw.circle(self.screen, color, (x, y), 10)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 10, 2)
            
            id_text = self.small_font.render(str(responder.id), True, (0, 0, 0))
            self.screen.blit(id_text, id_text.get_rect(center=(x, y)))
            
            if responder.carried_injured or responder.walking_followers:
                load_text = f"[{len(responder.carried_injured)}I|{len(responder.walking_followers)}W]"
                load_surf = self.tiny_font.render(load_text, True, (255, 0, 0))
                self.screen.blit(load_surf, (x - 18, y - 18))    
    def _draw_path_3d(self, path: List[Position3D], current_pos: Position3D, x_offset: int, y_offsets: List[int]):
        """Draw 3D path (changes color when crossing floors)"""
        if not path:
            return
        
        points_by_floor = {0: [], 1: []}
        
        # Add current position
        curr_floor = current_pos.floor
        curr_x = int(x_offset + current_pos.col * self.cell_size + self.cell_size // 2)
        curr_y = int(y_offsets[curr_floor] + current_pos.row * self.cell_size + self.cell_size // 2)
        points_by_floor[curr_floor].append((curr_x, curr_y))
        
        # Add path points
        prev_floor = curr_floor
        for pos in path:
            floor = pos.floor
            px = int(x_offset + pos.col * self.cell_size + self.cell_size // 2)
            py = int(y_offsets[floor] + pos.row * self.cell_size + self.cell_size // 2)
            
            if floor != prev_floor:
                # Floor transition - mark with special indicator
                prev_x = int(x_offset + pos.col * self.cell_size + self.cell_size // 2)
                prev_y = int(y_offsets[prev_floor] + pos.row * self.cell_size + self.cell_size // 2)
                pygame.draw.circle(self.screen, Config.COLORS['stair_path'], (prev_x, prev_y), 6)
                pygame.draw.circle(self.screen, Config.COLORS['stair_path'], (px, py), 6)
            
            points_by_floor[floor].append((px, py))
            prev_floor = floor
        
        # Draw path segments per floor
        for floor in [0, 1]:
            if len(points_by_floor[floor]) >= 2:
                color = Config.COLORS['evac_path'] if prev_floor == 0 else Config.COLORS['path']
                pygame.draw.lines(self.screen, color, False, points_by_floor[floor], 3)
    
    def _draw_info_panel(self, panel_x: int):
        """Draw information panel"""
        panel_y = 30
        
        # Background
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (panel_x, 0, self.panel_width, self.window_height))
        
        # Title
        title = self.font.render("EVACUATION", True, (0, 0, 0))
        self.screen.blit(title, (panel_x + 10, panel_y))
        panel_y += 20
        
        # Fast-forward indicator
        if not self.responders_arrived:
            ff_text = f"FAST-FORWARDING..."
            ff_surf = self.font.render(ff_text, True, (255, 100, 0))
            self.screen.blit(ff_surf, (panel_x + 10, panel_y))
            panel_y += 30
            
            progress = (self.current_time / Config.RESPONSE_TIME) * 100
            progress_text = f"Response: {progress:.1f}%"
            progress_surf = self.small_font.render(progress_text, True, (200, 80, 0))
            self.screen.blit(progress_surf, (panel_x + 10, panel_y))
            panel_y += 25
        
        # Time
        time_text = f"Time: {self.current_time:.1f}s"
        self.screen.blit(self.small_font.render(time_text, True, (0, 0, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 20
        
        hazard_text = f"Hazard Tick: {self.hazard_sim.current_tick}"
        self.screen.blit(self.tiny_font.render(hazard_text, True, (80, 80, 80)), 
                        (panel_x + 10, panel_y))
        panel_y += 20
        
        # Occupants
        rescued_text = f"Rescued: {self.stats['rescued_occupants']}/{self.stats['total_occupants']}"
        self.screen.blit(self.small_font.render(rescued_text, True, (0, 150, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 20
        
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
        
        panel_y += 5
        
        # Rooms
        rooms_text = f"Rooms: {self.stats['rooms_cleared']}/{self.stats['total_rooms']}"
        self.screen.blit(self.small_font.render(rooms_text, True, (0, 0, 150)), 
                        (panel_x + 10, panel_y))
        panel_y += 10
        
        # Responders
        self.screen.blit(self.small_font.render("RESPONDERS:", True, (0, 0, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 10
        
        for responder in self.responders:
            state_colors = {
                ResponderState.IDLE: (60, 60, 60),
                ResponderState.MOVING_TO_ROOM: (200, 150, 0),
                ResponderState.USING_STAIRS: (255, 100, 255),
                ResponderState.SEARCHING_ROOM: (0, 180, 255),
                ResponderState.EVACUATING: (0, 200, 0)
            }
            state_color = state_colors.get(responder.state, (0, 0, 0))
            
            state_text = f"R{responder.id}: {responder.state.value.upper()}"
            text_surf = self.small_font.render(state_text, True, state_color)
            shadow_surf = self.small_font.render(state_text, True, (200, 200, 200))
            self.screen.blit(shadow_surf, (panel_x + 16, panel_y + 1))
            self.screen.blit(text_surf, (panel_x + 15, panel_y))
            panel_y += 20
            
            # Floor info
            floor_text = f"  Floor {responder.position.floor + 1}"
            floor_surf = self.tiny_font.render(floor_text, True, (100, 100, 100))
            self.screen.blit(floor_surf, (panel_x + 20, panel_y))
            panel_y += 10
            
            if responder.assigned_room is not None:
                room = self.floor_plan.rooms[responder.assigned_room]
                di_seconds = room.time_until_danger * Config.SECONDS_PER_HAZARD_TICK
                room_text = f"  Rm{responder.assigned_room} (F{room.floor + 1}) D_i:{di_seconds:.0f}s"
                room_surf = self.small_font.render(room_text, True, (40, 40, 40))
                self.screen.blit(room_surf, (panel_x + 20, panel_y))
                panel_y += 10
            
            load_text = f"  Load: {len(responder.carried_injured)}I+{len(responder.walking_followers)}W"
            load_surf = self.small_font.render(load_text, True, (0, 0, 0))
            self.screen.blit(load_surf, (panel_x + 20, panel_y))
            panel_y += 10
            
            saved_text = f"  Saved: {responder.total_occupants_rescued}"
            saved_surf = self.small_font.render(saved_text, True, (0, 100, 0))
            self.screen.blit(saved_surf, (panel_x + 20, panel_y))
            panel_y += 10
        
        # Hazard info
        panel_y += 10
        self.screen.blit(self.small_font.render("HAZARD:", True, (0, 0, 0)), 
                        (panel_x + 10, panel_y))
        panel_y += 10
        
        for floor in range(self.floor_plan.num_floors):
            fire_cells = np.sum(self.hazard_sim.fire_intensity[floor] > 0.01)
            fire_text = f"Floor {floor + 1} Fire: {fire_cells}"
            self.screen.blit(self.tiny_font.render(fire_text, True, (200, 0, 0)), 
                            (panel_x + 15, panel_y))
            panel_y += 10
        
        if self.simulation_complete:
            panel_y += 15
            self.screen.blit(self.font.render("COMPLETE!", True, (0, 150, 0)), 
                           (panel_x + 10, panel_y))
    
    def _print_final_statistics(self):
        """Print final statistics"""
        print("\n" + "="*80)
        print("FINAL STATISTICS (TWO-FLOOR)")
        print("="*80)
        print(f"Simulation Time: {self.current_time:.2f}s")
        print(f"Hazard Ticks: {self.hazard_sim.current_tick}")
        print(f"\nOccupants:")
        print(f"  Total: {self.stats['total_occupants']}")
        print(f"  Rescued: {self.stats['rescued_occupants']}")
        success_rate = 100 * self.stats['rescued_occupants'] / max(1, self.stats['total_occupants'])
        print(f"  Success: {success_rate:.1f}%")
        
        # Per-floor statistics
        for floor in range(self.floor_plan.num_floors):
            floor_rescued = sum(1 for occ in self.occupant_manager.occupants.values() 
                              if occ.position.floor == floor and occ.state == OccupantState.RESCUED)
            floor_total = sum(1 for occ in self.occupant_manager.occupants.values() 
                             if occ.position.floor == floor)
            print(f"\n  Floor {floor + 1}:")
            print(f"    Total: {floor_total}")
            print(f"    Rescued: {floor_rescued}")
        
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
    
    print("\n" + "="*80)
    print("TWO-FLOOR EMERGENCY EVACUATION SIMULATION")
    print("Multi-Story Building with Vertical Fire/Smoke Spread")
    print("="*80)
    print("\nFeatures:")
    print("  - Two floors with staircases")
    print("  - Vertical fire and smoke spread")
    print("  - 3D pathfinding with stair traversal")
    print("  - Fire starts on Floor 2, evacuate to Floor 1 exits")
    print("  - Hazard timing: 1 tick = 6 seconds")
    print("  - Monte Carlo D_i calculation (multi-floor)")
    print("  - Dynamic responder calculation")
    print("  - Stair climb time: 15 seconds per floor")
    print("\nControls:")
    print("  ESC - Exit simulation")
    print("="*80 + "\n")
    
    sim = TwoFloorEvacuationSimulation(
        num_responders=None,
        occupant_density=0.15,
        fire_start_floor=1,  # Floor 2
        fire_start_position=(3, 3)
    )
    
    sim.run()


if __name__ == "__main__":
    main()
