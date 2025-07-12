from components.Entity import Entity
from components.button import Button
from components.label import Label
from components.sprite import Sprite




def new_game():
    from core.engine import engine
    engine.switch_to("Play")
    
def quit_game():
    from core.engine import engine
    engine.running=False
    
def menu():
    Entity(Sprite("main-menu background pic.png",50, is_ui=True))
    
    new_game_button=Entity(Label("main/EBGaramond-Regular.ttf", "New Game", 80, (255,255,255)))
    quit_game_button=Entity(Label("main/EBGaramond-Regular.ttf", "Quit Game", 80, (255,255,255)))
    
    new_button_size= new_game_button.get(Label).get_bounds()
    quit_button_size=quit_game_button.get(Label).get_bounds()
    
    from core.camera import camera
    new_game_button.x=camera.width/2 - new_button_size.width/2
    new_game_button.y=camera.height-350 
    quit_game_button.x=camera.width/2 - quit_button_size.width/2
    quit_game_button.y=camera.height-200
    
    new_game_button.add(Button(new_game, new_button_size))
    quit_game_button.add(Button(quit_game, quit_button_size))