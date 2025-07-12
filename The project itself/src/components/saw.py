from core.engine import engine
from components.physics import Body, get_bodies_within_circle
from components.player import Player
from components.enemy_ai import EnemyAI
from components.combat import Combat
from components.sprite import Sprite

class Saw:
    """Handles saw behavior and damage dealing"""
    def __init__(self):
        self.damage = 25
        self.damage_cooldown = 0
        self.damage_cooldown_max = 30  # Half second at 60 FPS
        self.damage_radius = 14
        
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
    
    def update(self):
        # Handle damage cooldown
        if self.damage_cooldown > 0:
            self.damage_cooldown -= 1
        
        # Check for entities to damage
        if self.damage_cooldown <= 0:
            self.check_damage()
    
    def check_damage(self):
    # Get entities in damage radius
        nearby = get_bodies_within_circle(
        self.entity.x + 16,
        self.entity.y + 16,
        self.damage_radius
        )
    
        damaged_something = False
        
        for body in nearby:
            # Add AgentAI to the check
            from components.agent_ai import AgentAI
            if body.entity.has(Player) or body.entity.has(EnemyAI) or body.entity.has(AgentAI):
                target_combat = body.entity.get(Combat)
                if target_combat:
                    target_combat.take_damage(self.damage)
                    damaged_something = True
                    print(f"Saw damaged entity, health remaining: {target_combat.health}")  # Debug print
        
        if damaged_something:
            self.damage_cooldown = self.damage_cooldown_max
        
        def breakdown(self):
            if self in self.engine.active_objs:
                self.engine.active_objs.remove(self)

class SawPath:
    """Handles saw movement along a defined path"""
    def __init__(self, path_points):
        self.points = path_points  # List of (x,y) coordinates defining the path
        self.current_point = 0  # Current target point index
        self.move_speed = 2  # Pixels per frame
        self.direction = 1  # 1 for forward, -1 for backward
        
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
    
    def update(self):
        if not hasattr(self, 'entity'):
            return
            
        next_x, next_y = self.get_next_position(self.entity.x, self.entity.y)
        self.entity.x = next_x
        self.entity.y = next_y
    
    def get_next_position(self, current_x, current_y):
        """Calculate next position along path"""
        target = self.points[self.current_point]
        
        # Calculate direction to next point
        dx = target[0] - current_x
        dy = target[1] - current_y
        
        # Move in straight lines (either horizontally or vertically)
        if abs(dx) >= abs(dy):  # Moving horizontally
            if dx > 0:
                return current_x + self.move_speed, current_y
            elif dx < 0:
                return current_x - self.move_speed, current_y
        else:  # Moving vertically
            if dy > 0:
                return current_x, current_y + self.move_speed
            elif dy < 0:
                return current_x, current_y - self.move_speed
        
        # If we've reached the target point (or very close)
        if abs(dx) < self.move_speed and abs(dy) < self.move_speed:
            self.current_point += self.direction
            # Reverse direction if we hit end of path
            if self.current_point >= len(self.points) or self.current_point < 0:
                self.direction *= -1
                self.current_point += self.direction * 2
            
            return current_x, current_y
        
        return current_x, current_y
    
    def breakdown(self):
        if self in self.engine.active_objs:
            self.engine.active_objs.remove(self)