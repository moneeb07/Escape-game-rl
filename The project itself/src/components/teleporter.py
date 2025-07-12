


from core.area import area
import pygame

def teleport(area_file, target_x, target_y, entity):
    """Teleport any valid entity to new area and position"""
    from core.engine import engine
    
    if area_file == "next_area.map":  # Or whatever your final area trigger is
        engine.complete_level()  # Signal level completion
    else:
        from core.area import area
        area.load_file(area_file)
        if target_x is not None and target_y is not None:
            entity.x = target_x * 16
            entity.y = target_y * 16

from components.physics import Trigger
class Teleporter(Trigger):
    def __init__(self, area_file, target_x=None, target_y=None, x=0, y=0, width=48, height=48):
        def teleport_entity(other):
            # Check if the entity is either Player or AgentAI
            from components.agent_ai import AgentAI
            from components.player import Player
            if other.has(Player) or other.has(AgentAI):
                teleport(area_file, int(target_x), int(target_y), other)
                
        super().__init__(teleport_entity, x, y, width, height)
        print(f"Teleporter set at ({x}, {y}) with width {width} and height {height}")
