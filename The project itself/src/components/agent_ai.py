import math
import random
from components.combat import Combat
from components.enemy_ai import EnemyAI
from components.pathfinding import PathFinder
from components.physics import Body
from components.puzzle_components import Door, LeverState


class AgentAI:
    def __init__(self):
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
        
        # Known puzzle solution and lever positions
        self.lever_positions = [(4,3), (21,3), (4,13), (21,13)]
        from stages.play import puzzle
        self.solution = puzzle.target_sequence
        self.current_lever_index = 0
        
        # Movement
        self.current_path = None
        self.path_index = 0
        self.move_speed = 3
        self.retry_count = 0
        self.max_retries = 3
        self.collision_count = 0
        self.max_collisions = 5
        self.combat=None
        self.path_recalc_cooldown = 0
        self.path_recalc_delay = 60  # Wait 1 second between path recalcs
        self.last_enemy_pos = None   # Track enemy movement
        self.stuck_counter = 0       # Track if we're stuck
        # State
        self.state = "VERIFY_LEVERS"
        self.interaction_range = 60
        self.interaction_delay = 0
        
        print("Agent AI initialized!")
        self.setup_pathfinding()
        self.setup_combat()
    def setup_combat(self):
        """Initialize combat component"""
        if not self.combat and hasattr(self, 'entity'):
            print("Setting up agent combat")
            combat = Combat(100, self.on_agent_death)
            combat.damage = 20
            combat.attack_range = 50
            self.entity.add(combat)
            self.combat = combat
            self.last_health = combat.health  # Store initial health
            print("Agent combat initialized with health:", self.combat.health)

    def on_agent_death(self, entity):
        """Handle what happens when agent dies"""
        print("Agent has died!")
        from core.engine import engine
        engine.switch_to("Play")  # Restart level

    def setup_pathfinding(self):
        """Initialize pathfinder with current map state"""
        from core.area import area
        
        
        enemy_pos = None
        for entity in area.entities:
            if entity.has(EnemyAI):
                enemy_pos = (entity.x, entity.y)
                break
        
        self.pathfinder = PathFinder(area.map, enemy_pos)
        print("Pathfinding setup complete!")

    def verify_lever_states(self):
        """More robust lever state verification"""
        from core.area import area
        incorrect_levers = []
        
        print("\nChecking all lever states:")
        for i, pos in enumerate(self.lever_positions):
            found_lever = False
            for entity in area.entities:
                lever_state = entity.get(LeverState)
                if lever_state and entity.x // 32 == pos[0] and entity.y // 32 == pos[1]:
                    found_lever = True
                    current = lever_state.is_activated
                    desired = self.solution[i]
                    print(f"Lever {i} at {pos}: Current={current}, Desired={desired}")
                    if current != desired:
                        incorrect_levers.append((i, entity))
                    break
                    
            if not found_lever:
                print(f"Warning: Could not find lever at position {pos}")
                
        print(f"Found {len(incorrect_levers)} incorrect levers")
        return incorrect_levers

    def find_next_lever_to_fix(self, incorrect_levers):
        """Determine which incorrect lever to fix next"""
        if not incorrect_levers:
            return None
        
        closest_index = incorrect_levers[0][0]
        min_distance = float('inf')
        
        for index, entity in incorrect_levers:
            dx = entity.x - self.entity.x
            dy = entity.y - self.entity.y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = index
        
        return closest_index
    def handle_stuck(self):
        """Enhanced stuck handling"""
        print("Handling stuck situation...")
        
        # Clear current path
        self.current_path = None
        self.collision_count = 0
        
        body = self.entity.get(Body)
        if not body:
            return
            
        # Try increasingly larger movements to escape
        distances = [32, 48, 64, 96]
        directions = [
            (1, 0), (-1, 0),     # Horizontal
            (0, 1), (0, -1),     # Vertical
            (1, 1), (-1, 1),     # Diagonal
            (1, -1), (-1, -1)    # Diagonal
        ]
        
        for distance in distances:
            for dx, dy in directions:
                # Store original position
                original_x = self.entity.x
                original_y = self.entity.y
                
                # Try new position
                test_x = original_x + (dx * distance)
                test_y = original_y + (dy * distance)
                
                self.entity.x = test_x
                self.entity.y = test_y
                
                if body.is_position_valid():
                    print(f"Successfully escaped to ({test_x//32}, {test_y//32})")
                    return True
                
                # Reset position if invalid
                self.entity.x = original_x
                self.entity.y = original_y
        
        # If all else fails, try to find path to any known safe position
        safe_positions = [
            (5, 5),    # Top left
            (20, 5),   # Top right
            (5, 12),   # Bottom left
            (20, 12),  # Bottom right
            (12, 8)    # Center
        ]
        
        for pos_x, pos_y in safe_positions:
            if self.pathfinder.is_walkable(pos_x, pos_y):
                path = self.pathfinder.find_path(
                    (self.entity.x, self.entity.y),
                    (pos_x * 32, pos_y * 32)
                )
                if path:
                    self.current_path = path
                    self.path_index = 0
                    print(f"Found path to safe position ({pos_x}, {pos_y})")
                    return True
        
        # Last resort: reset to spawn
        print("No safe position found, resetting to spawn...")
        self.entity.x = 5 * 32
        self.entity.y = 5 * 32
        return False

    def move_along_path(self):
        """Move along current path with fixed speed"""
        if not self.current_path or self.path_index >= len(self.current_path):
            return True

        target = self.current_path[self.path_index]
        dx = target[0] - self.entity.x
        dy = target[1] - self.entity.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # If very close to target, move to next waypoint
        if distance < self.move_speed:
            self.path_index += 1
            return self.path_index >= len(self.current_path)
        
        # Calculate movement with exact speed
        move_x = (dx / distance) * self.move_speed
        move_y = (dy / distance) * self.move_speed
        
        # Store previous position
        prev_x = self.entity.x
        prev_y = self.entity.y
        
        # Try to move
        self.entity.x += move_x
        self.entity.y += move_y
        
        # Validate movement
        body = self.entity.get(Body)
        if body and not body.is_position_valid():
            self.entity.x = prev_x
            self.entity.y = prev_y
            self.collision_count += 1
            
            if self.collision_count >= self.max_collisions:
                print("Too many collisions, finding new path...")
                self.current_path = None
                self.collision_count = 0
                return False
        else:
            self.collision_count = 0  # Reset on successful movement
        
        return False
    def is_near_position(self, target_x, target_y):
        """Check if agent is near enough to interact"""
        dx = self.entity.x - target_x
        dy = self.entity.y - target_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < self.interaction_range

    def interact_with_lever(self):
        """Improved lever interaction with strict distance checking"""
        from core.area import area
        target_pos = self.lever_positions[self.current_lever_index]
        target_x = target_pos[0] * 32
        target_y = target_pos[1] * 32
        
        # First check if we're close enough to interact
        dx = target_x - self.entity.x
        dy = target_y - self.entity.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 48:  # Strict interaction range
            print(f"Too far from lever to interact. Distance: {distance}")
            return False
            
        for entity in area.entities:
            lever_state = entity.get(LeverState)
            if lever_state and entity.x // 32 == target_pos[0] and entity.y // 32 == target_pos[1]:
                print(f"Found lever {self.current_lever_index}")
                print(f"Current state: {lever_state.is_activated}")
                print(f"Desired state: {self.solution[self.current_lever_index]}")
                
                if lever_state.is_activated != self.solution[self.current_lever_index]:
                    lever_state.toggle()
                    print(f"Toggled lever {self.current_lever_index}")
                
                return True
        
        return False
    def is_good_escape_position(self, x, y, enemy_x, enemy_y):
        """Check if a position is good for escape (away from enemy and walls)"""
        # Must be walkable
        if not self.pathfinder.is_walkable(x, y):
            return False

        # Check distance from enemy (in tiles)
        distance_to_enemy = math.sqrt((x - enemy_x)**2 + (y - enemy_y)**2)
        if distance_to_enemy < 5:  # Must be at least 5 tiles away from enemy
            return False

        # Check surrounding tiles for walls
        wall_count = 0
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            check_x = x + dx
            check_y = y + dy
            if not self.pathfinder.is_walkable(check_x, check_y):
                wall_count += 1

        # Position is bad if too many surrounding walls
        return wall_count <= 2  # Allow at most 2 adjacent walls

    def check_danger(self):
        """Enhanced danger checking with delayed escape"""
        if self.path_recalc_cooldown > 0:
            self.path_recalc_cooldown -= 1
            return False

        # Check for health decrease first
        if self.combat:
            current_health = self.combat.health
            if current_health < self.last_health:
                print("Taking damage! Need immediate escape!")
                self.last_health = current_health
                self.path_recalc_cooldown = self.path_recalc_delay
                return True
                
        # Check for enemy proximity
        from core.area import area
        from components.enemy_ai import EnemyAI
        
        for entity in area.entities:
            if entity.has(EnemyAI):
                dx = entity.x - self.entity.x
                dy = entity.y - self.entity.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                enemy_ai = entity.get(EnemyAI)
                danger_range = enemy_ai.aggro_range   # Allow closer approach before escaping
                
                if distance < danger_range:
                    print(f"Enemy too close! Distance: {distance}, Danger range: {danger_range}")
                    self.path_recalc_cooldown = 15  # Quick updates when close
                    return True
                    
                if distance < enemy_ai.chase_range:
                    # Random chance to trigger escape when in chase range
                    if random.random() < 0.1:  # 10% chance per update to try escaping
                        print("Attempting preemptive escape!")
                        return True
        
        return False

    def update(self):
        """Updated update with strict lever interaction"""
        if not self.combat:
            self.setup_combat()
        if not hasattr(self, 'entity'):
            return
                
        if self.interaction_delay > 0:
            self.interaction_delay -= 1
            return
        if self.state == "MOVE_TO_EXIT":
            if not self.current_path:
                door_position = (12, 1)
                print("Finding path to door")
                self.current_path = self.pathfinder.find_path(
                    (self.entity.x, self.entity.y),
                    (door_position[0] * 32, door_position[1] * 32)
                )
                if self.current_path:
                    self.path_index = 0
            
            if self.current_path:
                if self.move_along_path():
                    # Simply check if we're near the door to teleport
                    door_x = 12 * 32
                    door_y = 0 * 32
                    dx = self.entity.x - door_x
                    dy = self.entity.y - door_y
                    distance = math.sqrt(dx * dx + dy * dy)
                    
                    if distance < 60:
                        from components.teleporter import teleport
                        teleport("next_area.map", 5, 5, self.entity)
                        print("Teleporting to next area")
                        #import pygame
                        # pygame.quit()
                        return
        # Check for enemy danger first
        in_danger = self.check_danger()
        if in_danger:
            print("Enemy nearby! Finding escape route...")
            new_path = self.try_alternative_path()
            if new_path:
                self.current_path = new_path
                self.path_index = 0
            else:
                self.emergency_escape()
            return
        
        # If not in danger, proceed with normal behavior
        if self.state == "VERIFY_LEVERS":
            incorrect_levers = self.verify_lever_states()
            if incorrect_levers:
                next_lever = self.find_next_lever_to_fix(incorrect_levers)
                if next_lever is not None:
                    self.current_lever_index = next_lever
                    self.current_path = None
                    self.state = "MOVE_TO_LEVER"
            else:
                print("Moving to door")
                self.state = "MOVE_TO_EXIT"
                self.current_path = None
                
        elif self.state == "MOVE_TO_LEVER":
            if not self.current_path:
                target_pos = self.lever_positions[self.current_lever_index]
                print(f"Finding path to lever {self.current_lever_index} at position {target_pos}")
                self.current_path = self.pathfinder.find_path(
                    (self.entity.x, self.entity.y),
                    (target_pos[0] * 32, target_pos[1] * 32)
                )
                if self.current_path:
                    self.path_index = 0
            
            if self.current_path:
                target_pos = self.lever_positions[self.current_lever_index]
                target_x = target_pos[0] * 32
                target_y = target_pos[1] * 32
                dx = target_x - self.entity.x
                dy = target_y - self.entity.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < 48:  # Only try to interact when very close
                    if self.interact_with_lever():
                        self.state = "VERIFY_LEVERS"
                        self.current_path = None
                        self.interaction_delay = 30  # Add delay after successful interaction
                        return
            
            # if self.move_along_path():
            #     if distance < 48:  # One final interaction attempt when path ends
            #         if self.interact_with_lever():
            #             self.state = "VERIFY_LEVERS"
            #             self.current_path = None
            #             self.interaction_delay = 30
            #             return
            #     self.state = "VERIFY_LEVERS"
            #     self.current_path = None
            
            if self.current_path:
                target_pos = self.lever_positions[self.current_lever_index]
                target_x = target_pos[0] * 32
                target_y = target_pos[1] * 32
                dx = target_x - self.entity.x
                dy = target_y - self.entity.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < 48:  # If close enough to lever
                    if self.interact_with_lever():
                        print("Successfully interacted with lever")
                        self.state = "VERIFY_LEVERS"
                        self.current_path = None
                        self.interaction_delay = 30  # Add delay after interaction
                        return
                
                if self.move_along_path():
                    print("Reached end of path to lever")
                    if not self.interact_with_lever():  # Try one final interaction
                        print("Could not interact with lever at end of path")
                    self.state = "VERIFY_LEVERS"
                    self.current_path = None
                    
        
    def breakdown(self):
        if self in self.engine.active_objs:
            self.engine.active_objs.remove(self)
    def move_to_exit(self):
        """Handle movement to exit door"""
        door_position = (12, 0)  # Door position from your map
        
        if not self.current_path:
            self.current_path = self.pathfinder.find_path(
                (self.entity.x, self.entity.y),
                (door_position[0] * 32, door_position[1] * 32)
            )
            if self.current_path:
                self.path_index = 0
        
        if self.move_along_path():
            # Check if the door is actually unlocked before trying to teleport
            from core.area import area
            for entity in area.entities:
                if entity.get(Door):
                    door = entity.get(Door)
                    if not door.locked and self.is_near_position(entity.x, entity.y):
                        # Only teleport if the door is unlocked and we're close enough
                        from components.teleporter import Teleporter, teleport
                        teleport("next_area.map", 5, 5, self.entity)
                        print("Teleported through unlocked door")
                        return True
                    elif door.locked:
                        print("Door is still locked! Cannot teleport!")
                        self.state = "VERIFY_LEVERS"  # Go back to checking levers
                        return False
        return False

        
        
    def interact_with_door(self):
        """Interact with exit door"""
        from core.area import area
        for entity in area.entities:
            if entity.get(Door):
                door_x = entity.x
                door_y = entity.y
                # Check if we're close enough to the door
                if self.is_near_position(door_x, door_y):
                    if not entity.get(Door).locked:
                        from components.teleporter import Teleporter
                        # Door is unlocked, trigger teleporter
                        if entity.has(Teleporter):
                            # Force trigger the teleporter
                            
                            print(self.solution)
                            teleporter = entity.get(Teleporter)
                            teleporter.on(self.entity)  # Pass the agent entity to trigger
                            print("Triggered teleporter")
                            return True
        return False
    def is_near_position(self, target_x, target_y):
        """Check if agent is physically near enough to interact"""
        dx = self.entity.x - target_x
        dy = self.entity.y - target_y
        distance = math.sqrt(dx * dx + dy * dy)
        print(f"Distance to target: {distance}, Required range: {self.interaction_range}")
        return distance < 60  # Strict distance check
    def move_to_door(self):
        """Move to door when all levers are correct"""
        door_position = (12, 0)  # Door coordinates from your map
        
        if not self.current_path:
            self.current_path = self.pathfinder.find_path(
                (self.entity.x, self.entity.y),
                (door_position[0] * 32, door_position[1] * 32)
            )
            if self.current_path:
                self.path_index = 0
                print("Found path to door")
            else:
                print("No path to door found")
        return self.move_along_path()
    def check_danger(self):
        """Simple check if in enemy's aggro range"""
        if self.path_recalc_cooldown > 0:
            self.path_recalc_cooldown -= 1
            return False

        # Check for health decrease first
        if self.combat:
            current_health = self.combat.health
            if current_health < self.last_health:
                print("Taking damage! Need to escape!")
                self.last_health = current_health
                self.path_recalc_cooldown = self.path_recalc_delay
                return True
                
        # Check for enemy proximity
        from core.area import area
        
        
        for entity in area.entities:
            from components.enemy_ai import EnemyAI
            if entity.has(EnemyAI):
                dx = entity.x - self.entity.x
                dy = entity.y - self.entity.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                enemy_ai = entity.get(EnemyAI)
                if distance < enemy_ai.aggro_range-20:
                    print(f"In enemy aggro range! Distance: {distance}")
                    return True
        
        return False


        
        

        
    
    def try_emergency_path(self):
        """Try more extreme alternatives when stuck"""
        current_x = self.entity.x // 32
        current_y = self.entity.y // 32
        
        # Try to move to corners or far points
        emergency_points = [
            (5, 5),        # Top left area
            (20, 5),       # Top right area
            (5, 12),       # Bottom left area
            (20, 12),      # Bottom right area
            (12, 8),       # Center area
        ]
        
        for point in emergency_points:
            path = self.pathfinder.find_path(
                (self.entity.x, self.entity.y),
                (point[0] * 32, point[1] * 32)
            )
            if path:
                print("Found emergency escape path!")
                self.stuck_counter = 0
                return path
                
        return None
    def is_position_dangerous(self, x, y):
        """Check if a position is too close to enemy"""
        from core.area import area
        for entity in area.entities:
            if entity.has(EnemyAI):
                dx = entity.x - x
                dy = entity.y - y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < 80:  # Consider position dangerous if too close to enemy
                    return True
        return False
    def emergency_escape(self):
        """Simple emergency escape to spawn"""
        print("Emergency escape to spawn!")
        self.entity.x = 5 * 32
        self.entity.y = 5 * 32
    def try_safe_corner_escape(self, enemy_entity):
        """Fallback escape to safe corners"""
        safe_corners = [
            (5, 5),    # Top left
            (20, 5),   # Top right
            (5, 12),   # Bottom left
            (20, 12),  # Bottom right
            (12, 8),   # Center
        ]
        
        # Sort corners by distance from enemy
        enemy_x = int(enemy_entity.x // 32)
        enemy_y = int(enemy_entity.y // 32)
        
        safe_corners.sort(key=lambda pos: 
            (pos[0] - enemy_x) ** 2 + (pos[1] - enemy_y) ** 2,
            reverse=True
        )
        
        for corner in safe_corners:
            if self.pathfinder.is_walkable(corner[0], corner[1]):
                path = self.pathfinder.find_path(
                    (self.entity.x, self.entity.y),
                    (corner[0] * 32, corner[1] * 32)
                )
                if path:
                    print(f"Using safe corner escape to {corner}")
                    return path
        
        return None
    def try_alternative_path(self):
        """Find simple escape path outside aggro range"""
        from core.area import area
        from components.enemy_ai import EnemyAI
        
        # Get enemy position
        enemy_entity = None
        for entity in area.entities:
            if entity.has(EnemyAI):
                enemy_entity = entity
                break
        
        if not enemy_entity:
            return None

        # Move to nearest safe corner outside aggro range
        safe_corners = [
            (5, 5),    # Top left
            (20, 5),   # Top right
            (5, 12),   # Bottom left
            (20, 12)   # Bottom right
        ]
        
        # Sort corners by distance from enemy (prefer further corners)
        enemy_x = int(enemy_entity.x // 32)
        enemy_y = int(enemy_entity.y // 32)
        
        safe_corners.sort(key=lambda pos: 
            (pos[0] - enemy_x) ** 2 + (pos[1] - enemy_y) ** 2,
            reverse=True
        )
        
        for corner in safe_corners:
            if self.pathfinder.is_walkable(corner[0], corner[1]):
                path = self.pathfinder.find_path(
                    (self.entity.x, self.entity.y),
                    (corner[0] * 32, corner[1] * 32)
                )
                if path:
                    print(f"Escaping to safe corner at {corner}")
                    return path
        
        return None
