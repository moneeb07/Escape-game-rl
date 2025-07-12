import random
from components.sprite import Sprite, loaded
from components.teleporter import Teleporter
from core.area import Area
from data.tiletypes import tiles
from components.puzzle_components import Door, LeverState, PuzzleSystem
from data.objects import create_entity
from components.puzzle_components import Door, PuzzleSystem



# Usage in play.py or similar gameplay logic
# Assuming `door_entity` is the game object representing the door

puzzle = None
NUM_LEVERS = 4


def play():
    global puzzle 
    # Initialize the starting area
    random_solution = [random.choice([True, False]) for _ in range(NUM_LEVERS)]  # Example solution
    puzzle = PuzzleSystem.get_instance()
    puzzle.set_solution(random_solution)
    
    area = Area("start.map", tiles)
    print(f"Area initialized: {area}")
    
    # # Create enemy and saw paths
    enemy = create_entity(6, 13, 5)
    path1_start = create_entity(8, 8, 1, ["True"])
    path1_end = create_entity(10, 8, 15, ["True"])
    path2_start = create_entity(8, 16, 1, ["True"])
    path2_end = create_entity(10, 16, 15, ["True"])
    path3_start = create_entity(8, 1, 8, ["False"])
    path3_end = create_entity(10, 23, 8, ["False"])
    
    # Create saws
    path_points = [(50,192+64), (700,192+64)]
    saw = create_entity(7, 5, 8, [str(path_points)])
    path_points_2 = [(320-64, 48), (320-64, 486)]
    saw2 = create_entity(7, 8, 10, [str(path_points_2)])
    path_points_3 = [((320-64)+(32*8), 48), ((320-64)+(32*8), 486)]
    saw3 = create_entity(7, 16, 10, [str(path_points_3)])

    # Create levers
    lever_positions = [(4,3), (21, 3), (4, 13), (21,13)]
    levers = []
    for pos in lever_positions:
        lever = create_entity(4, pos[0], pos[1])
        levers.append(lever)
        puzzle.register_lever(lever)
        print(f"Created lever at ({lever.x}, {lever.y})")

    # Create the door
    door = create_entity(5, 12, 0)
    
    def on_puzzle_solved():
        """Single callback to handle puzzle completion"""
        print("\nPUZZLE SOLVED! Unlocking door...")
        
        # Update door sprite
        door_sprite = door.get(Sprite)
        if door_sprite:
            door_sprite.image = loaded["door_open.png"]
            print("Door sprite updated to open")
        
        # Unlock door
        door_component = door.get(Door)
        if door_component:
            door_component.locked = False
            print("Door unlocked")
        
        # Add teleporter
        if not door.has(Teleporter):
            door.add(Teleporter(area_file="next_area.map", target_x=5, target_y=5))
            print("Teleporter added to door")
        
        print("Door fully unlocked and ready for teleport")
        return True

    # Set single callback
    puzzle.set_callback(on_puzzle_solved)
    print(f"Puzzle system initialized with solution: {random_solution}")
