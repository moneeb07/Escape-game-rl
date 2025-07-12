from core.engine import engine
import pygame
class HealthBar:
    def __init__(self, combat_component, offset_y=-20):
        self.combat = combat_component
        self.offset_y = offset_y
        self.width = 32
        self.height = 4
        # Get engine instance properly
        from core.engine import engine
        self.engine = engine
        self.engine.ui_drawables.append(self)
    
    def draw(self, screen):
        from core.camera import camera
        
        # Get screen position
        x = self.combat.entity.x - camera.x
        y = self.combat.entity.y + self.offset_y - camera.y
        
        # Draw background (red)
        bg_rect = pygame.Rect(x, y, self.width, self.height)
        pygame.draw.rect(screen, (255, 0, 0), bg_rect)
        
        # Draw health (green)
        health_percentage = self.combat.health / self.combat.max_health
        health_width = int(self.width * health_percentage)
        if health_width > 0:
            health_rect = pygame.Rect(x, y, health_width, self.height)
            pygame.draw.rect(screen, (0, 255, 0), health_rect)
    
    def breakdown(self):
        if self in self.engine.ui_drawables:
            self.engine.ui_drawables.remove(self)
            
class Combat:
    def __init__(self, health, on_death):
        self.health = health
        self.max_health = health
        self.cooldown = 0
        self.damage = 10
        self.attack_range = 50
        self.on_death = on_death
        
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
        self.health_bar = None
    
    def setup_health_bar(self):
        if hasattr(self, 'entity') and not self.health_bar:
            self.health_bar = HealthBar(self)
    
    def breakdown(self):
        if self in self.engine.active_objs:
            self.engine.active_objs.remove(self)
        if self.health_bar:
            self.health_bar.breakdown()
    
    def take_damage(self, amount):
        prev_health = self.health
        self.health = max(0, self.health - amount)  # Prevent negative health
        actual_damage = prev_health - self.health
        
        print(f"Taking {actual_damage} damage!")
        print(f"Health before: {prev_health}, after: {self.health}")
        
        if self.health <= 0:
            self.on_death(self.entity)
    
    def attack(self, target_entity):
        if self.cooldown <= 0:
            target_combat = target_entity.get(Combat)
            if target_combat:
                print(f"Attacking! Damage: {self.damage}")
                print(f"Target health before: {target_combat.health}")
                target_combat.take_damage(self.damage)
                print(f"Target health after: {target_combat.health}")
                self.cooldown = 60  # 1 second cooldown
                return True  # Attack successful
        return False  # Attack failed or on cooldown
    
    def can_attack(self):
        return self.cooldown <= 0
    
    def is_in_range(self, target_x, target_y):
        dx = target_x - self.entity.x
        dy = target_y - self.entity.y
        distance = (dx * dx + dy * dy) ** 0.5
        return distance <= self.attack_range
    
    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1
            
        if not self.health_bar:
            self.setup_health_bar()