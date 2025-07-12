from components.Entity import Entity
from components.sprite import Sprite
import pygame
from core.engine import engine

class HealthBar:
    def __init__(self, combat_component, offset_y=-20):
        self.combat = combat_component
        self.offset_y = offset_y
        self.width = 32
        self.height = 4
        self.border = 1
        engine.ui_drawables.append(self)
    
    def breakdown(self):
        engine.ui_drawables.remove(self)
    
    def draw(self, screen):
        from core.camera import camera
        
        # Get screen position
        x = self.combat.entity.x - camera.x
        y = self.combat.entity.y + self.offset_y - camera.y
        
        # Draw border
        border_rect = pygame.Rect(x - 1, y - 1, self.width + 2, self.height + 2)
        pygame.draw.rect(screen, (0, 0, 0), border_rect)
        
        # Draw background
        bg_rect = pygame.Rect(x, y, self.width, self.height)
        pygame.draw.rect(screen, (255, 0, 0), bg_rect)
        
        # Draw health
        health_percentage = self.combat.health / self.combat.max_health
        health_width = int(self.width * health_percentage)
        if health_width > 0:
            health_rect = pygame.Rect(x, y, health_width, self.height)
            pygame.draw.rect(screen, (0, 255, 0), health_rect)

class DamageNumber:
    def __init__(self, x, y, amount, color=(255, 0, 0)):
        self.x = x
        self.y = y
        self.amount = amount
        self.color = color
        self.lifetime = 60  # 1 second at 60 FPS
        self.float_speed = 1
        self.font = pygame.font.Font(None, 24)
        self.text = self.font.render(str(amount), True, self.color)
        engine.ui_drawables.append(self)
        engine.active_objs.append(self)
    
    def update(self):
        self.y -= self.float_speed
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.breakdown()
    
    def breakdown(self):
        engine.ui_drawables.remove(self)
        engine.active_objs.remove(self)
    
    def draw(self, screen):
        from core.camera import camera
        screen_x = self.x - camera.x
        screen_y = self.y - camera.y
        screen.blit(self.text, (screen_x, screen_y))