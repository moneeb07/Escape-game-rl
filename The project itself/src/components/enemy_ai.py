import math
import random

from components.puzzle_components import LeverState
from core.engine import engine
from components.physics import get_bodies_within_circle, Body
from components.player import Player
from components.combat import Combat

def on_enemy_death(entity):
    """Handle enemy death more robustly"""
    try:
        # Try standard area removal first
        from core.area import area
        if entity in area.entities:
            area.remove_entity(entity)
        else:
            # If not in area.entities, clean up components manually
            for component in entity.components:
                if hasattr(component, 'breakdown'):
                    component.breakdown()
            # Clear components after breakdown
            entity.components.clear()
        print("Enemy removed successfully")
    except Exception as e:
        print(f"Error during enemy cleanup: {e}")
        # Ensure components are cleaned up even if area removal fails
        if hasattr(entity, 'components'):
            for component in entity.components:
                if hasattr(component, 'breakdown'):
                    component.breakdown()
            entity.components.clear()


class EnemyAI:
    def __init__(self):
        self.target = None
        self.targeted_entity = None
        self.vision_range = 300      # Range to detect players
        self.aggro_range = 100       # Increased range to start chase
        self.chase_range = 60   
        self.walk_speed = 15
        self.attack_range = 32
        self.combat = None
        self.move_delay = 0
        self.move_delay_max = 10
        self.in_chase = False      
        # New AI state variables
        self.current_target_lever = None # How far to look for levers
        self.state = "chase_player"   # Initial state
        self.state_switch_delay = 0   # Delay before switching states
        # New aggro range - only chase within this distance
        # Puzzle-related variables
        self.current_target_lever = None
        self.lever_check_range = 150
        self.state = "find_lever"  # Changed default state to focus on puzzle
        self.state_switch_delay = 0
        self.known_lever_positions = [(4,3), (21,3), (4,13), (21,13)]  # Same as agent's positions
        self.current_lever_index = 0
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
        print("Enemy AI initialized")
    
    def is_target_in_range(self, target_entity=None, range_value=None):
        """Check if target is within specified range"""
        entity_to_check = target_entity or self.targeted_entity
        if not entity_to_check:
            return False
            
        dx = entity_to_check.x - self.entity.x
        dy = entity_to_check.y - self.entity.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        range_to_check = range_value or (self.chase_range if self.in_chase else self.aggro_range)
        return distance <= range_to_check
    
    def find_next_lever(self):
        """Find the next lever to check/toggle"""
        from core.area import area
        
        for pos in self.known_lever_positions:
            for entity in area.entities:
                lever_state = entity.get(LeverState)
                if lever_state and entity.x // 32 == pos[0] and entity.y // 32 == pos[1]:
                    # Enemy wants to deactivate levers that are activated
                    if lever_state.is_activated:
                        self.current_target_lever = entity
                        self.target = (entity.x, entity.y)
                        return True
        return False
    def find_active_lever(self):
        """Look for activated levers within range"""
        seen_objects = get_bodies_within_circle(
            self.entity.x, 
            self.entity.y, 
            self.lever_check_range
        )
        
        for body in seen_objects:
            lever_state = body.entity.get(LeverState)
            if lever_state and lever_state.is_activated:
                print("Found activated lever!")
                return body.entity
        return None
    
    def decide_action(self):
        """Decide next action with more aggressive chase behavior"""
        # Update target information
        target_found = self.find_target()
        
        # If we're chasing or found a target in aggro range
        if target_found or (self.in_chase and self.targeted_entity):
            if self.is_target_in_range(range_value=self.chase_range):
                if self.state != "chase_player":
                    print("Starting chase!")
                self.state = "chase_player"
                return
        
        # Only stop chasing if target is well out of range
        if self.state == "chase_player":
            if not self.targeted_entity or not self.is_target_in_range(range_value=self.chase_range):
                print("Target escaped! Returning to lever duty")
                self.state = "find_lever"
                self.in_chase = False
                
        # Handle lever state when not chasing
        if self.state == "find_lever":
            if not self.current_target_lever or self.is_near_lever(self.current_target_lever):
                if not self.find_next_lever():
                    self.patrol_lever_positions()

    def is_near_lever(self, lever):
        """Check if we're close enough to interact with a lever"""
        dx = self.entity.x - lever.x
        dy = self.entity.y - lever.y
        distance = (dx * dx + dy * dy) ** 0.5
        return distance < 40  # Adjust interaction range as needed
    def setup_combat(self):
        """Initialize combat component with proper values"""
        if not self.combat and hasattr(self, 'entity'):
            print("Setting up enemy combat")
            combat = Combat(100, on_enemy_death)
            combat.damage = 25  # Damage per hit
            combat.attack_range = 40  # Slightly larger than aggro range
            self.entity.add(combat)
            self.combat = combat
            print(f"Enemy combat initialized with damage: {combat.damage}, range: {combat.attack_range}")

    def patrol_lever_positions(self):
        """Patrol between lever positions when no immediate tasks"""
        if not hasattr(self, 'patrol_index'):
            self.patrol_index = 0
        
        pos = self.known_lever_positions[self.patrol_index]
        self.target = (pos[0] * 32, pos[1] * 32)
        
        # If we're close to current patrol point, move to next
        dx = self.entity.x - (pos[0] * 32)
        dy = self.entity.y - (pos[1] * 32)
        if math.sqrt(dx * dx + dy * dy) < 50:
            self.patrol_index = (self.patrol_index + 1) % len(self.known_lever_positions)

    
    def breakdown(self):
        """Ensure proper cleanup"""
        try:
            from core.engine import engine
            if self in engine.active_objs:
                engine.active_objs.remove(self)
            print("Enemy AI cleaned up")
        except Exception as e:
            print(f"Error during enemy AI cleanup: {e}")
    
    def find_target(self):
        """Look for targets within vision range"""
        seen_objects = get_bodies_within_circle(
            self.entity.x, 
            self.entity.y, 
            self.vision_range
        )
        
        closest_target = None
        min_distance = float('inf')
        
        for body in seen_objects:
            from components.agent_ai import AgentAI
            if body.entity.has(Player) or body.entity.has(AgentAI):
                dx = body.entity.x - self.entity.x
                dy = body.entity.y - self.entity.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_target = body.entity
        
        # Check if target is within appropriate range
        if closest_target:
            range_to_check = self.chase_range if self.in_chase else self.aggro_range
            if min_distance <= range_to_check:
                self.targeted_entity = closest_target
                self.target = (closest_target.x, closest_target.y)
                self.in_chase = True
                return True
            
        if min_distance > self.chase_range:
            self.in_chase = False
            
        return False
    
    def move_to_target(self):
        """Move enemy towards the target position with delay"""
        if not self.target:
            return
            
        # Only move if delay timer is 0
        if self.move_delay > 0:
            self.move_delay -= 1
            return
            
        self.move_delay = self.move_delay_max  # Reset delay
            
        body = self.entity.get(Body)
        prev_x = self.entity.x
        prev_y = self.entity.y
        
        # Calculate movement direction
        dx = self.target[0] - self.entity.x
        dy = self.target[1] - self.entity.y
        
        # Normalize movement to make diagonal movement same speed
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > 0:
            dx = dx / distance * self.walk_speed
            dy = dy / distance * self.walk_speed
        
        # Apply movement
        self.entity.x += dx
        if not body.is_position_valid():
            self.entity.x = prev_x
            
        self.entity.y += dy
        # if not body.is_position_valid():
        #     self.entity.y = prev_y
    
    def try_attack_target(self):
        """Attack target if in range"""
        if not self.targeted_entity or not self.combat:
            return False
            
        dx = self.targeted_entity.x - self.entity.x
        dy = self.targeted_entity.y - self.entity.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance <= self.attack_range:
            if self.targeted_entity.has(Combat):
                print("Attacking target!")
                return self.combat.attack(self.targeted_entity)
        return False

        
    def update(self):
        """Main update loop with more aggressive behavior"""
        if not self.combat:
            self.setup_combat()
        
        self.decide_action()
        
        if self.state == "chase_player":
            if self.target:
                self.move_to_target()
                self.try_attack_target()
        if self.state == "find_lever":
            if self.target:
                self.move_to_target()
                if self.current_target_lever and self.is_near_lever(self.current_target_lever):
                    lever_state = self.current_target_lever.get(LeverState)
                    if lever_state and lever_state.is_activated:
                        lever_state.toggle()
                        self.current_target_lever = None
    