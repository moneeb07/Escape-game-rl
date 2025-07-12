# pathfinding.py
from core.map import TileKind
import heapq
import math

class Node:
    def __init__(self, x, y, g_cost=0, h_cost=0):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # Cost from start
        self.h_cost = h_cost  # Estimated cost to goal
        self.f_cost = g_cost + h_cost
        self.parent = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class PathFinder:
    def __init__(self, game_map, saw_paths, enemy_pos=None):
        self.map = game_map
        self.saw_paths = saw_paths
        self.enemy_pos = enemy_pos
        print("PathFinder initialized")
    
    # In pathfinding.py
    def is_walkable(self, x, y):
        """Check if a position is walkable and safe from saws"""
        # Bounds check
        if (x < 0 or y < 0 or 
            y >= len(self.map.tiles) or 
            x >= len(self.map.tiles[0])):
            return False
        
        # Check tile type
        tile = self.map.tiles[y][x]
        if tile in [2, 3, 4]:  # Wall tiles
            return False

        
        return True
    
    def get_neighbors(self, current):
        """Get valid neighboring nodes"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            new_x = current.x + dx
            new_y = current.y + dy
            
            if self.is_walkable(new_x, new_y):
                movement_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                neighbor = Node(new_x, new_y)
                neighbor.g_cost = current.g_cost + movement_cost
                neighbors.append(neighbor)
                print(f"Added neighbor at ({new_x}, {new_y})")
        
        return neighbors

    def calculate_h_cost(self, node, goal):
        """Calculate heuristic cost to goal"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        return (dx + dy) * 1.1  # Slight bias towards goal

    def find_path(self, start_pos, end_pos):
        """Find path using A* pathfinding"""
        # Convert to grid coordinates
        start_x, start_y = int(start_pos[0] // 32), int(start_pos[1] // 32)
        end_x, end_y = int(end_pos[0] // 32), int(end_pos[1] // 32)
        
        print(f"Finding path from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        
        if not self.is_walkable(start_x, start_y) or not self.is_walkable(end_x, end_y):
            print("Start or end position is not walkable!")
            return None
        
        # Initialize start and goal nodes
        start = Node(start_x, start_y)
        goal = Node(end_x, end_y)
        
        open_set = []
        closed_set = set()
        
        # Add start node
        start.h_cost = self.calculate_h_cost(start, goal)
        start.f_cost = start.h_cost
        heapq.heappush(open_set, start)
        
        while open_set:
            current = heapq.heappop(open_set)
            
            if (current.x, current.y) == (goal.x, goal.y):
                # Path found, reconstruct it
                path = []
                while current:
                    path.append((current.x * 32 + 16, current.y * 32 + 16))
                    current = current.parent
                print(f"Path found with {len(path)} nodes")
                return path[::-1]
            
            if (current.x, current.y) in closed_set:
                continue
                
            closed_set.add((current.x, current.y))
            
            # Process neighbors
            for neighbor in self.get_neighbors(current):
                if (neighbor.x, neighbor.y) in closed_set:
                    continue
                
                neighbor.h_cost = self.calculate_h_cost(neighbor, goal)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current
                heapq.heappush(open_set, neighbor)
        
        print("No path found!")
        return None