import pygame
from core.camera import camera


image_path = "content/sprites"

loaded = {}

class Sprite:
    def __init__(self, image, scale, is_ui=False):
        from core.engine import engine
        self.entity = None  # Initialize entity reference
        
        if image in loaded:
            self.image = loaded[image]
        else:
            self.image = pygame.image.load(image_path + "/" + image)
            self.image = pygame.transform.scale(self.image, (16*scale, 16*scale))
            loaded[image] = self.image
            
        engine.drawables.append(self)
        if is_ui:
            engine.ui_drawables.append(self)
        else:
            engine.drawables.append(self)
        self.is_ui = is_ui
    def breakdown(self):
        from core.engine import engine
        engine.drawables.remove(self)

    def draw(self, screen):
        if not hasattr(self, 'entity') or self.entity is None:
            return  # Skip drawing if no entity reference
            
        pos = (self.entity.x - camera.x, self.entity.y - camera.y) \
                if not self.is_ui else \
                (self.entity.x, self.entity.y)
        screen.blit(self.image, pos)

# Load things uniquely.
