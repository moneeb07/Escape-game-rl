import pygame
pygame.init()

engine = None
default_width = 800
default_height = 600

class Engine:
    def __init__(self, game_title) -> None:
        from core.camera import create_screen
        global engine
        engine = self

        self.active_objs = [] # Anything with an update() method which can be called
        self.background_drawables = []
        self.drawables = [] # Anything to be drawn in the world
        self.ui_drawables = [] # Anything to be drawn over the world
        self.level_complete = False  # Add completion tracking
        self.clear_color = (23,17,26) # Default color if nothing else is drawn somewhere
        self.screen = create_screen(default_width, default_height, game_title) # The rectangle in the window itself
        self.stages = {}
        self.current_stage = None
        self.step = 0  # Add frame counter

    def register(self, stage_name, func):
        self.stages[stage_name] = func

    def switch_to(self, stage_name):
        self.reset()
        self.current_stage = stage_name 
        func = self.stages[stage_name]
        print(f"Switching to {self.current_stage}")
        func()

    def run(self):
        from core.input import keys_down

        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    keys_down.add(event.key)
                elif event.type == pygame.KEYUP:
                    keys_down.remove(event.key)

            # Update Code
            for a in self.active_objs:
                a.update()

            # Draw Code
            self.screen.fill(self.clear_color)
            
            # Draw background items like the tiles
            for b in self.background_drawables:
                b.draw(self.screen)

            # Draw the main objects
            for s in self.drawables:
                s.draw(self.screen)

            # Draw UI Stuff
            for l in self.ui_drawables:
                l.draw(self.screen)

            pygame.display.flip()

            # Increment frame counter
            self.step += 1

            # Cap the frames
            pygame.time.delay(17)
                
        pygame.quit()
        return self.level_complete
    
    
    def complete_level(self):
        """Mark level as complete and stop game loop"""
        self.level_complete = True
        self.running = False
    def reset(self):
        from components.physics import reset_physics
        reset_physics()
        self.active_objs.clear()
        self.drawables.clear()
        self.ui_drawables.clear()
        self.background_drawables.clear()
        self.step = 0  # Reset step counter