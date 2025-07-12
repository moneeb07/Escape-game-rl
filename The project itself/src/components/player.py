import pygame
from components.sprite import Sprite
from core.input import is_key_pressed
from components.Entity import Entity
from components.physics import Body, triggers
from components.label import Label
from components.inventory import Inventory
from components.ui.inventory_view import InventoryView
from components.combat import Combat

movement_speed = 2

def player_death(entity):
    print("Player died!")
    from core.engine import engine
    engine.switch_to('Menu')

class Player:
    def __init__(self):
        from core.area import area
        from core.engine import engine
        engine.active_objs.append(self)
        
        # Combat setup
        self.combat = None
        
        # Location label
        self.loc_label = Entity(Label("main/EBGaramond-Regular.ttf", 
                                    "X: 0 - Y: 0")).get(Label)
        self.area_label = Entity(Label("main/EBGaramond-Regular.ttf", 
                                    area.name)).get(Label)
        
        from core.camera import camera
        self.loc_label.entity.y = camera.height - 50
        self.loc_label.entity.x = 10
        self.area_label.entity.x = 10

    def setup_combat(self):
        """Initialize combat component"""
        if not self.combat and hasattr(self, 'entity'):
            print("Setting up player combat")
            combat = Combat(100, player_death)  # Player starts with 100 HP
            combat.damage = 20  # Player deals 20 damage
            combat.attack_range = 50  # 50 pixel attack range
            self.entity.add(combat)
            self.combat = combat
            print("Player combat initialized with health:", combat.health)

    def update(self):
        # Ensure combat is set up
        if not self.combat:
            self.setup_combat()
            
        # Position update for UI
        self.loc_label.set_text(f"X:{self.entity.x} - Y: {self.entity.y}")
        
        # Store previous position for collision checking
        previous_x = self.entity.x
        previous_y = self.entity.y
        
        # Get components
        sprite = self.entity.get(Sprite)
        body = self.entity.get(Body)
        
        # Movement
        if is_key_pressed(pygame.K_w):
            self.entity.y -= movement_speed
        if is_key_pressed(pygame.K_s):
            self.entity.y += movement_speed
        if not body.is_position_valid():
            self.entity.y = previous_y

        if is_key_pressed(pygame.K_a):
            self.entity.x -= movement_speed
        if is_key_pressed(pygame.K_d):
            self.entity.x += movement_speed
        if not body.is_position_valid():
            self.entity.x = previous_x
            
        # Combat controls
        if is_key_pressed(pygame.K_SPACE):  # Spacebar to attack
            self.try_attack()
            
        # Camera update
        from core.camera import camera
        camera.x = self.entity.x - camera.width/2 + sprite.image.get_width()/2
        camera.y = self.entity.y - camera.height/2 + sprite.image.get_height()/2

        # Check triggers
        for t in triggers:
            if body.is_colliding_with(t):
                t.on(self.entity)
                
    def try_attack(self):
        """Attempt to attack nearby enemies"""
        if not self.combat or not self.combat.can_attack():
            return
            
        # Look for enemies in range
        from components.enemy_ai import EnemyAI
        from components.physics import get_bodies_within_circle
        
        nearby = get_bodies_within_circle(
            self.entity.x,
            self.entity.y,
            self.combat.attack_range
        )
        
        # Attack first enemy in range
        for body in nearby:
            if body.entity.has(EnemyAI):
                enemy_combat = body.entity.get(Combat)
                if enemy_combat:
                    self.combat.attack(enemy_combat)
                    break  # Only attack one enemy at a time