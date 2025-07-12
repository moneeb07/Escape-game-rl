# import pygame
# from core.input import keys_down, is_key_pressed

# from components.sprite import sprites, Sprite #the list of sprites in sprite.py
# from core.map import Map, TileKind
# from core.camera import create_screen
# from components.Entity import Entity,active_objs
# from components.physics import Body
# from core.area import Area, area

# pygame.init()
# #win=create_screen(800,600,"Second level")
# win=pygame.display.set_mode((800,600))
# pygame.display.set_caption("Second level")

# clear_color=(23, 17, 26)
# running=True
# # Sprite("The project itself/Radiant Dungeon asset sheet/No-Background/Interactables.png",0,80,16,16,3,48*3,48*3)
# # player=Player("The project itself/Radiant Dungeon asset sheet/No-Background/Hero.png",0,0,16,16, 400, 300,3)
# #player=Entity(Player(), Sprite("Hero.png",0,0,16,16,3),Body(8,20,16,16),x=32*11,y=32*7)
# from data.tiletypes import tiles

# # def make_lever(x,y):
# #     Entity(Sprite("Interactables.png",0,80,16,16,3),x=x,y=y)

# # make_lever(48*5,48*5)
    

# area=Area("start.map",tiles)
# #Game loop
# while running:
#     for event in pygame.event.get():
#         if event.type==pygame.QUIT:
#             running=False
#         elif event.type==pygame.KEYDOWN: #Check if the event occouring in the game is a key being pressed
#             keys_down.add(event.key) #Adding the keys in the keys_down set in input.py file if key is pressed
#         elif event.type ==pygame.KEYUP:
#             keys_down.remove(event.key) #Removed the keys in keys_down set in input.py if key is released
    
#     #Update codes
#     for a in active_objs:
#         a.update()
    
#     #Draw code     
#     win.fill(clear_color)
#     area.map.draw(win)
#     for s in sprites:
#         s.draw(win) #It draws the sprite per frame and the display is updated continuously to make it feel like it is moving. It will do it for all sprites in the sprites list.
#     #UI stuff
    
#     pygame.display.flip()
#     pygame.time.delay(17)
# pygame.quit()sd

import pygame



def main():
    from core.engine import Engine
    from stages.menu import menu
    from stages.play import play
    e = Engine("Escape room")
    e.register("Menu", menu)
    e.register("Play", play)
    e.switch_to("Play")
    return e.run()  # Returns True if level completed, False if quit

  # Clean shutdown



if __name__ == "__main__":
    main()