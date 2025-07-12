keys_down=set() #keep track of all keys pressed, set stores unique values only
import pygame

def is_key_pressed(key):
    return key in keys_down #should return true or false


def is_mouse_pressed(button):
    return pygame.mouse.get_pressed()[button]
