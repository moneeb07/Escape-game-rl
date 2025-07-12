import pygame
from components.sprite import Sprite
from core.engine import engine

class FlashEffect:
    def __init__(self, entity, flash_color=(255, 0, 0), duration=10):
        self.entity = entity
        self.sprite = entity.get(Sprite)
        if not self.sprite:
            return
        
        self.original_image = self.sprite.image.copy()
        self.flash_color = flash_color
        self.duration = duration
        self.time_left = duration
        
        engine.active_objs.append(self)
        self.apply_flash()
    
    def apply_flash(self):
        """Apply flash effect to sprite"""
        if not self.sprite:
            return
            
        flash_surface = pygame.Surface(self.sprite.image.get_size()).convert_alpha()
        flash_surface.fill(self.flash_color)
        
        flashed_image = self.original_image.copy()
        flashed_image.blit(flash_surface, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        
        self.sprite.image = flashed_image
    
    def update(self):
        """Update flash effect"""
        self.time_left -= 1
        
        if self.time_left <= 0:
            self.breakdown()
            
    def breakdown(self):
        """Clean up effect"""
        if self.sprite:
            self.sprite.image = self.original_image
        if self in engine.active_objs:
            engine.active_objs.remove(self)