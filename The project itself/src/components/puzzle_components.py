from components.Entity import Entity
from components.sprite import Sprite, loaded
from components.physics import Body, Trigger
from core.input import is_key_pressed
import pygame
from components.player import Player
class LeverState:
    def __init__(self):
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
        self.is_activated = False
        self._ensure_images_loaded()
        
    def _ensure_images_loaded(self):
        """Make sure lever and door images are loaded"""
        image_path = "content/sprites/"
        if "lever_up.png" not in loaded:
            up_image = pygame.image.load(image_path + "lever_up.png")
            loaded["lever_up.png"] = pygame.transform.scale(up_image, (16*2, 16*2))
        if "lever_down.png" not in loaded:
            down_image = pygame.image.load(image_path + "lever_down.png")
            loaded["lever_down.png"] = pygame.transform.scale(down_image, (16*2, 16*2))
        if "door_closed.png" not in loaded:
            closed_image = pygame.image.load(image_path + "door_closed.png")
            loaded["door_closed.png"] = pygame.transform.scale(closed_image, (16*2, 16*2))
        if "door_open.png" not in loaded:
            open_image = pygame.image.load(image_path + "door_open.png")
            loaded["door_open.png"] = pygame.transform.scale(open_image, (16*2, 16*2)) 
        
    def update(self):
        if not hasattr(self, 'sprite_initialized'):
            self.sprite = self.entity.get(Sprite)
            self.sprite_initialized = True
        
    def toggle(self):
        self.is_activated = not self.is_activated
        if hasattr(self, 'sprite_initialized') and self.sprite:
            self.sprite.image = loaded["lever_down.png" if self.is_activated else "lever_up.png"]
        puzzle_system = PuzzleSystem.get_instance()
        if puzzle_system:
            puzzle_system.check_solution()
    
    def breakdown(self):
        if self.engine:
            self.engine.active_objs.remove(self)

class LeverTrigger(Trigger):
    def __init__(self):
        super().__init__(self.on_trigger, 0, 0, 32, 32)
        self.key_was_pressed = False  # Track previous key state

    def on_trigger(self, other):
        """Toggle lever state only once per key press when colliding."""
        from components.puzzle_components import LeverState

        if other.has(Player):  # Ensure the colliding entity is the player
            if is_key_pressed(pygame.K_e):  # Check if 'E' is pressed
                if not self.key_was_pressed:  # Ensure it's a new press
                    print("E key pressed! Toggling lever.")
                    lever_state = self.entity.get(LeverState)
                    if lever_state:
                        lever_state.toggle()
                    self.key_was_pressed = True  # Prevent further toggling until key release
            else:
                self.key_was_pressed = False  # Reset when key is releasede


class PuzzleSystem:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = PuzzleSystem()
        return cls._instance
    
    def __init__(self):
        self.levers = []
        self.target_sequence = []  # Initialize empty sequence
        self.on_solved = None
        print("PuzzleSystem initialized")
    
    def register_lever(self, lever):
        """Register a lever with the puzzle system"""
        if lever not in self.levers:
            self.levers.append(lever)
            print(f"Registered lever. Total levers: {len(self.levers)}")
    
    def set_solution(self, sequence):
        """Set the target sequence of lever states"""
        self.target_sequence = sequence
        print(f"Solution set to: {self.target_sequence}")
    
    def set_callback(self, callback):
        """Set the function to call when puzzle is solved"""
        self.on_solved = callback
        print("Puzzle callback set")
    
    def check_solution(self):
        """Check if current lever states match the solution"""
        if not self.target_sequence or not self.levers:
            print("No solution or levers set!")
            return False
            
        current_sequence = [lever.get(LeverState).is_activated for lever in self.levers]
        print(f"Checking solution: Current={current_sequence}, Target={self.target_sequence}")
        
        if current_sequence == self.target_sequence:
            if self.on_solved:
                print("Solution correct! Triggering callback.")
                self.on_solved()
                return True
        return False
            


class Door:
    def __init__(self):
        from core.engine import engine
        self.engine = engine
        self.engine.active_objs.append(self)
        self.locked = True
        self.sprite = None

    def unlock(self):
        """Unlock the door and ensure sprite changes"""
        if self.locked:
            print("Door unlocked!")
            self.locked = False
            if not hasattr(self, 'sprite_initialized'):
                self.update()
            if self.sprite:
                self.sprite.image = loaded["door_open.png"]
                print("Door sprite updated to open")

    def update(self):
        """Ensure sprite is initialized and reflects correct state"""
        if not hasattr(self, 'sprite_initialized'):
            self.sprite = self.entity.get(Sprite)
            self.sprite_initialized = True
            # Ensure sprite matches lock state
            if self.sprite and not self.locked:
                self.sprite.image = loaded["door_open.png"]