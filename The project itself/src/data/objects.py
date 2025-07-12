from components.Entity import Entity


from components.combat import Combat
from components.enemy_ai import EnemyAI
from components.saw import Saw, SawPath
from components.sprite import Sprite
from components.player import Player
from components.physics import Body
from components.teleporter import Teleporter
from components.inventory import Inventory, DroppedItem
from components.puzzle_components import LeverState, LeverTrigger,Door
from core import area, engine
from data.item_types import item_types
from components.agent_ai import AgentAI
def determine_orientation(path_points):
    """Determine if path is vertical based on points"""
    points = eval(path_points)
    start, end = points[0], points[1]
    dy = abs(end[1] - start[1])
    dx = abs(end[0] - start[0])
    return dy > dx
def agent_death(entity):
    print("Agent died!")
    from core.engine import engine
    engine.switch_to("Play")  # Restart level
entity_factories=[
    #0- Make player
    lambda args: Entity(Player(), Sprite("Hero.png",2,False), Body(8,8,16,16)),
    
    #1- Make Skulls
    lambda args: Entity( Sprite("Skulls.png",2)),
    
    #2- Make teleporter
    lambda args: Entity(Teleporter(area_file="next_area.map", player_x=5, player_y=5), Sprite("door_open.png", 2)),
    
    #3- Make coins as dropped item
    lambda args: Entity( DroppedItem(item_types[int(args[0])],int(args[1])), Sprite(item_types[int(args[0])].icon_name)),
    #4- Make lever
    lambda args: Entity(
        Sprite("lever_up.png", 2),  # Create sprite first
        LeverState(),               # Then add state
        LeverTrigger(),
        Body(8,8,16,16)# Finally add trigger
    ),
    # 5 - Make locked door
    lambda args: Entity(
    Sprite("door_closed.png", 2),  # Closed door sprite initially
     Door(),  
    Body(8,8,32,32)# Door behavior
    ),
    #6- Make Enemy
    lambda args: Entity(
        Sprite("Enemy.png", 2),     # Enemy sprite
        Body(8, 8, 8, 8),         # Collision body
        EnemyAI()                   # Enemy behavior
    ),
    #7- Make Saw1
    lambda args: Entity(
        Sprite("saw_vertical.png" if determine_orientation(args[0]) else "saw_horizontal.png", 2),
        Body(8, 8, 24, 24),
        Saw(),  # Pass is_vertical flag
        SawPath(eval(args[0]))
    ),
    lambda args: Entity(
        Sprite("saw_path_start_v.png" if eval(args[0]) else "saw_path_start_h.png", 2)
    ),
    #9- Path Middle
    lambda args: Entity(
        Sprite("saw_path_middle_v.png" if eval(args[0]) else "saw_path_middle_h.png", 2)
    ),
    #10- Path End
    lambda args: Entity(
        Sprite("saw_path_end_v.png" if eval(args[0]) else "saw_path_end_h.png", 2)
    ),
    #-11 Entity AI
    lambda args: Entity(
    Sprite("Hero.png", 2),     # Use player sprite or different one
    Body(8, 8, 12, 12),        # Same collision as player
    AgentAI()                  # Agent behavior
)
]

def create_entity(id, x,y,data=None):
    factory=entity_factories[id]
    e=factory(data)
    e.x=x*32
    e.y=y*32
    from core.area import area
    if area:
        area.entities.append(e)
    return e