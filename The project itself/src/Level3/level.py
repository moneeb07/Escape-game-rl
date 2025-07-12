import pygame
import sys
import time
import heapq
import random
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 27
WALL_COLOR = (0, 0, 0)
PATH_COLOR = (255, 255, 255)
FIRE_COLOR = (255, 69, 0)
VISITED_COLOR = (173, 216, 230)
LIFE_BAR_COLOR = (34, 139, 34)  # Green
AGENT_COLOR = (0, 0, 255)
ADVERSARY_COLOR = (255, 0, 0)
TRAP_SIZE = 20  # Size of the trap image (adjust based on your sprite size)
INITIAL_HEALTH = 100
FIRE_DAMAGE = 25
TRAP_DAMAGE = 10
STUN_DURATION = 2
player_health = INITIAL_HEALTH

agent_stunned = False  # Tracks whether the agent is stunned
stun_counter = 1

# 20x20 maze with multiple paths (0=wall, 1=path, 2=fire)
MAZE = [
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
]

# Window setup
ROWS = len(MAZE)
COLS = len(MAZE[0])
STATS_WIDTH = 400  # Adjust to the new desired width of the stats window
MAZE_OFFSET = STATS_WIDTH  # The maze starts after the stats window
WINDOW_SIZE = (COLS * CELL_SIZE + STATS_WIDTH, ROWS * CELL_SIZE)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Escape Room Level 3 with Adversary")

# Load and scale images
# Load the images (Make sure these images exist in your working directory)
path_img = pygame.image.load('c:/VScode projects/AI project/The project itself/src/Level3/assets/path.png')
wall_img = pygame.image.load('c:/VScode projects/AI project/The project itself/src/Level3/assets/wall.png')

# Load images for the traces
trace_agent_img = pygame.image.load("c:/VScode projects/AI project/The project itself/src/Level3/assets/tracea.png")  # Image for agent trace
trace_adversary_img = pygame.image.load("c:/VScode projects/AI project/The project itself/src/Level3/assets/traceo.png")  # Image for adversary trace

# Resize images if needed (to fit cell size)
trace_agent_img = pygame.transform.scale(trace_agent_img, (CELL_SIZE, CELL_SIZE))
trace_adversary_img = pygame.transform.scale(trace_adversary_img, (CELL_SIZE, CELL_SIZE))

# Optionally, resize the images to match the size of the cells
path_img = pygame.transform.scale(path_img, (CELL_SIZE, CELL_SIZE))
wall_img = pygame.transform.scale(wall_img, (CELL_SIZE, CELL_SIZE))

player_img = pygame.image.load('c:/VScode projects/AI project/The project itself/src/Level3/assets/Hero.png')
exit_img = pygame.image.load('c:/VScode projects/AI project/The project itself/src/Level3/assets/exit.png')
fire_img = pygame.image.load('c:/VScode projects/AI project/The project itself/src/Level3/assets/fire.png')
adversary_img = pygame.image.load('c:/VScode projects/AI project/The project itself/src/Level3/assets/adversary.png')
trap_img = pygame.image.load("c:/VScode projects/AI project/The project itself/src/Level3/assets/trap.png")  # Load the trap image (make sure the image exists)
trap_img = pygame.transform.scale(trap_img, (TRAP_SIZE, TRAP_SIZE))  # Resize the image to fit in the grid

player_img = pygame.transform.scale(player_img, (CELL_SIZE - 4, CELL_SIZE - 4))
exit_img = pygame.transform.scale(exit_img, (CELL_SIZE - 4, CELL_SIZE - 4))
fire_img = pygame.transform.scale(fire_img, (CELL_SIZE - 4, CELL_SIZE - 4))
adversary_img = pygame.transform.scale(adversary_img, (CELL_SIZE - 4, CELL_SIZE - 4))

# Code to preload sounds
pygame.mixer.init()
background_music = "c:/VScode projects/AI project/The project itself/src/Level3/assets/game.mp3"
fire_sound = "c:/VScode projects/AI project/The project itself/src/Level3/assets/fire.mp3"
trap_sound = "c:/VScode projects/AI project/The project itself/src/Level3/assets/trap.mp3"
win_sound = "c:/VScode projects/AI project/The project itself/src/Level3/assets/win.wav"
gameover_sound = "c:/VScode projects/AI project/The project itself/src/Level3/assets/gameover.wav"

# Load sounds
pygame.mixer.music.load(background_music)  # Background music
pygame.mixer.music.play(-1)  # Play in a loop
fire_effect = pygame.mixer.Sound(fire_sound)
trap_effect = pygame.mixer.Sound(trap_sound)
win_effect = pygame.mixer.Sound(win_sound)
gameover_effect = pygame.mixer.Sound(gameover_sound)


START_POS = None  # Will be set in `initialize_positions`
ADVERSARY_START = None
END_POS = None


NUM_TRAPS = 15  # Adjust this number based on how many traps you want

def place_traps():
    """Place traps at random positions in the maze."""
    trap_positions = []

    for row in range(ROWS):
        for col in range(COLS):
            # Check if the current cell is a path (1) and not start/end position
            if MAZE[row][col] == 1 and (row, col) != (0, 0) and (row, col) != (ROWS-1, COLS-1):
                trap_positions.append((row, col))

    # Randomly select trap positions
    random.shuffle(trap_positions)
    trap_positions = trap_positions[:NUM_TRAPS]  # Select the first NUM_TRAPS positions

    for trap in trap_positions:
        row, col = trap
        MAZE[row][col] = 3  # Mark the cell as a trap


def get_neighbors(pos, fire_allowed=False):
    """Get valid neighbors, optionally allowing fire tiles."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = pos[0] + dx, pos[1] + dy
        if 0 <= x < ROWS and 0 <= y < COLS and MAZE[x][y] != 0:  # Not a wall
            if fire_allowed or MAZE[x][y] != 2:  # Fire tiles allowed if specified
                neighbors.append((x, y))
    return neighbors



def heuristic(pos, target):
    """Heuristic that prioritizes paths with fewer fire tiles."""
    manhattan_distance = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
    fire_count = sum(1 for neighbor in get_neighbors(pos, fire_allowed=True) if MAZE[neighbor[0]][neighbor[1]] == 2)
    return manhattan_distance + fire_count * 5  # Fire tiles are weighted higher.



# Function to place random fires
def place_random_fires(num_fires):
    """Randomly place fires on the maze (only on path cells)."""
    fire_positions = []
    for _ in range(num_fires):
        while True:
            row = random.randint(0, ROWS - 1)
            col = random.randint(0, COLS - 1)
            if MAZE[row][col] == 1 and (row, col) not in fire_positions:
                MAZE[row][col] = 2  # Set this cell to fire
                fire_positions.append((row, col))
                break
    return fire_positions

def check_fire_damage(pos, current_health):
    """Check if the agent is on fire and decrease health accordingly."""
    row, col = pos
    if MAZE[row][col] == 2:  # If there's fire in the cell
        current_health -= FIRE_DAMAGE
        print(f"Fire damage! Health remaining: {max(0, current_health)}")
    return max(0, current_health)

def check_trap_damage(pos, current_health):
    """Check if the player is on a trap and apply damage."""
    row, col = pos
    if MAZE[row][col] == 3:  # Check if the current position has a trap
        current_health -= TRAP_DAMAGE
        print(f"Trap encountered! Health remaining: {max(0, current_health)}")
    return max(0, current_health)

def update_health(player, current_pos):
    if MAZE[current_pos[0]][current_pos[1]] == 2:  # Fire
        player.health -= 10
        fire_effect.play()  # Play fire sound
    elif MAZE[current_pos[0]][current_pos[1]] == 3:  # Trap
        player.health -= 20
        trap_effect.play()  # Play trap sound

    # Additional health checks if necessary
    if player.health <= 0:
        gameover_effect.play()  # Play gameover sound
        pygame.time.delay(2000)  # Allow sound to play
        display_game_over("GAME OVER!")
        pygame.mixer.music.stop()
        pygame.quit()
        sys.exit()

def escape(player, current_pos):
    if MAZE[current_pos[0]][current_pos[1]] == 4:  # Exit point
        win_effect.play()  # Play win sound
        pygame.time.delay(2000)  # Allow sound to play
        display_game_over("PLAYER ESCAPED!")
        pygame.mixer.music.stop()
        pygame.quit()
        sys.exit()


def check_adversary_capture(player, adversary_pos):
    if player.pos == adversary_pos:
        gameover_effect.play()  # Play gameover sound
        pygame.time.delay(2000)  # Allow sound to play
        display_game_over("CAUGHT BY OPPONENT!")
        pygame.mixer.music.stop()
        pygame.quit()
        sys.exit()



# Game setup to add fires to the maze
NUM_FIRES = 50  # Number of fires to place in the maze



def a_star(start, target):
    """A* algorithm with fire and trap prioritization."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current, fire_allowed=True):
            fire_penalty = 25 if MAZE[neighbor[0]][neighbor[1]] == 2 else 0
            trap_penalty = TRAP_DAMAGE if MAZE[neighbor[0]][neighbor[1]] == 3 else 0  # Trap penalty
            total_penalty = fire_penalty + trap_penalty
            tentative_g_score = g_score[current] + 1 + total_penalty

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                came_from[neighbor] = current

    return []  # Return an empty path if no valid path exists.

def calculate_distance(pos1, pos2):
    """Calculate the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def agent_avoidance_path(agent_position, adversary_position, target_position):
    """Recalculate the agent's path to avoid the adversary."""
    open_set = []
    heapq.heappush(open_set, (0, agent_position))
    came_from = {agent_position: None}
    g_score = {agent_position: 0}
    f_score = {agent_position: heuristic(agent_position, target_position)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target_position:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current, fire_allowed=True):
            # Check if the neighbor brings the agent too close to the adversary
            if calculate_distance(neighbor, adversary_position) <= 2:  # Avoid positions too close to the adversary
                continue

            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target_position)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                came_from[neighbor] = current

    return []  # Return an empty path if no valid path found


def dijkstra(start, target, allow_hazards=True):
    """Dijkstra's algorithm for the adversary, considering hazards only when necessary."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current, fire_allowed = True):
            penalty = 0
            if allow_hazards:  # Only apply hazard penalties if allowed
                fire_penalty = 25 if MAZE[neighbor[0]][neighbor[1]] == 2 else 0
                trap_penalty = TRAP_DAMAGE if MAZE[neighbor[0]][neighbor[1]] == 3 else 0
                penalty = fire_penalty + trap_penalty
            tentative_g_score = g_score[current] + 1 + penalty

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                came_from[neighbor] = current

    return []  # Return an empty path if no valid path exists.


def draw_maze(offset=MAZE_OFFSET):
    """Draw the maze and environment using images for path and walls."""
    for row in range(ROWS):
        for col in range(COLS):
            x, y = col * CELL_SIZE + offset, row * CELL_SIZE

            if MAZE[row][col] == 1:  # Path
                screen.blit(path_img, (x, y))
            elif MAZE[row][col] == 2:  # Fire
                screen.blit(path_img, (x, y))
                screen.blit(fire_img, (x + 2, y + 2))
            elif MAZE[row][col] == 3:  # Trap
                screen.blit(path_img, (x, y))
                trap_x = x + (CELL_SIZE - TRAP_SIZE) // 2
                trap_y = y + (CELL_SIZE - TRAP_SIZE) // 2
                screen.blit(trap_img, (trap_x, trap_y))
            else:  # Wall
                screen.blit(wall_img, (x, y))


import pygame.font  # Ensure pygame.font is imported

def draw_stats():
    """Draw the stats window with a background image, health bar, and health number."""
    # Draw the background image for the stats window
    stats_background = pygame.image.load("c:/VScode projects/AI project/The project itself/src/Level3/assets/stats_background.png")
    stats_background = pygame.transform.scale(stats_background, (STATS_WIDTH, WINDOW_SIZE[1]))
    screen.blit(stats_background, (0, 0))

    # Load the custom font from your directory
    font_path = "c:/VScode projects/AI project/The project itself/src/Level3/assets/PressStart2P-Regular.ttf"
    font = pygame.font.Font(font_path, 30)  # Main font size 30 for health and labels

    # Color: Dark purple (almost black)
    dark_purple = (50, 0, 50)

    # Position offset to move everything to the right
    offset_x = 50  # Adjust this value to move everything to the right as needed

    # Display "Level 3" at the top of the stats window
    level_label = font.render("Level 3", True, dark_purple)  # Dark purple color for the label
    screen.blit(level_label, (30 + offset_x, 80))  # Adjusted x position

    # Health label
    health_label = font.render("Health", True, dark_purple)  # Dark purple color for the label
    screen.blit(health_label, (10 + offset_x, WINDOW_SIZE[1] - 180))  # Adjusted x position

    # Increase the gap between the health label and health number
    health_number = font.render(f"{player_health}", True, dark_purple)  # Health number in dark purple
    screen.blit(health_number, (10 + offset_x, WINDOW_SIZE[1] - 120))  # Adjusted x position

    # Draw health bar as a single continuous image
    num_health_segments = player_health // 5  # Each image represents 5 health
    for i in range(num_health_segments):
        x_offset = 10 + offset_x + i * (HEALTH_SEGMENT_WIDTH)  # Adjusted x position for health segments
        y_position = WINDOW_SIZE[1] - 80  # Slightly above the bottom, adjusted for position
        screen.blit(health_images[i], (x_offset, y_position))



# Constants for health bar images
HEALTH_SEGMENT_WIDTH = 15  # Width of each health segment
HEALTH_SEGMENT_HEIGHT = 40  # Height of each health segment

# Preload health images and resize them
health_images = [
    pygame.transform.scale(pygame.image.load(f"c:/VScode projects/AI project/The project itself/src/Level3/assets/health{i}.png"), (HEALTH_SEGMENT_WIDTH, HEALTH_SEGMENT_HEIGHT))
    for i in range(1, 21)
]

def display_game_over(message):
    """Display the game over window with a message."""
    font_path = "c:/VScode projects/AI project/The project itself/src/Level3/assets/PressStart2P-Regular.ttf"
    font = pygame.font.Font(font_path, 50)  # Larger font size for game over message

    # Color: Dark purple (almost black)
    dark_purple = (50, 0, 50)

    # Render the message in dark purple
    game_over_text = font.render(message, True, dark_purple)

    # Create a surface for the game over screen
    game_over_surface = pygame.Surface((WINDOW_SIZE[0], WINDOW_SIZE[1]))
    game_over_surface.fill((0, 0, 0))  # Fill the screen with black to simulate a game over screen

    # Position the message in the center
    text_rect = game_over_text.get_rect(center=(WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2))
    
    # Blit the game over surface and the text onto the screen
    screen.blit(game_over_surface, (0, 0))
    screen.blit(game_over_text, text_rect)

    pygame.display.update()
def reset_maze():
    """Reset maze to initial state."""
    global MAZE
    MAZE = [
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    ]

def restart_game():
    """Reset game state and restart."""
    global player_health, agent_stunned, stun_counter
    player_health = INITIAL_HEALTH
    agent_stunned = False
    stun_counter = 1
    main()

def main():
      # Re-initialize pygame at start
    global player_health, agent_stunned, stun_counter
    clock = pygame.time.Clock()

    START_POS = (0, 0)  # Agent at top-left corner
    ADVERSARY_START = (ROWS - 2, COLS - 1)  # Adversary at top-right corner
    END_POS = (ROWS - 1, COLS - 1)  # Exit at bottom-right corner
    reset_maze()
    # Initialize hazards before pathfinding
    place_random_fires(NUM_FIRES)
    place_traps()  # Place traps in the maze

    solution_path = a_star(START_POS, END_POS)
    if not solution_path:
        print("No valid path found!")
        pygame.quit()
        sys.exit()

    agent_index = 0  # Start with the first position in the path
    adversary_path = []
    adversary_index = 0
    adversary_trace = []

    stun_counter = 0  # Initialize stun counter

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if agent_index >= len(solution_path) - 1:
            print("Agent has reached the exit!")
            win_effect.play()  # Play win sound
            display_game_over("PLAYER ESCAPED!")  # Call the game over function with the "escaped" message
            pygame.time.delay(2000)  # Allow sound to play
            pygame.quit()
            sys.exit()

        screen.fill(PATH_COLOR)

        # Draw the stats window on the left side
        draw_stats()

        # Draw maze shifted to the right
        draw_maze(MAZE_OFFSET)

        # Draw agent's visited path using image
        for pos in solution_path[:agent_index]:
            screen.blit(trace_agent_img, (pos[1] * CELL_SIZE + MAZE_OFFSET, pos[0] * CELL_SIZE))
        
        # Draw adversary's trace using image
        for pos in adversary_trace:
            screen.blit(trace_adversary_img, (pos[1] * CELL_SIZE + MAZE_OFFSET, pos[0] * CELL_SIZE))

        # Draw exit
        exit_x, exit_y = END_POS[1] * CELL_SIZE + MAZE_OFFSET + 2, END_POS[0] * CELL_SIZE + 2
        screen.blit(exit_img, (exit_x, exit_y))

        # Draw agent
        agent_x, agent_y = solution_path[agent_index][1] * CELL_SIZE + MAZE_OFFSET + 2, solution_path[agent_index][0] * CELL_SIZE + 2
        screen.blit(player_img, (agent_x, agent_y))

        # Adversary movement logic
        if adversary_index >= len(adversary_path) or adversary_path[adversary_index] != solution_path[agent_index]:
            adversary_path = dijkstra(
                ADVERSARY_START if adversary_index == 0 else adversary_path[adversary_index], 
                solution_path[agent_index], 
                allow_hazards=False
            )
            adversary_index = 0

        if adversary_index < len(adversary_path):
            current_adversary_pos = adversary_path[adversary_index]
            adversary_trace.append(current_adversary_pos)
            adv_x, adv_y = current_adversary_pos[1] * CELL_SIZE + MAZE_OFFSET + 2, current_adversary_pos[0] * CELL_SIZE + 2
            screen.blit(adversary_img, (adv_x, adv_y))

        pygame.display.update()

        if stun_counter == 0:  # Allow the agent to move only if not stunned
            current_pos = solution_path[agent_index]
            
            # Fire or trap check and play corresponding sound
            if MAZE[current_pos[0]][current_pos[1]] == 2:  # Fire
                fire_effect.play()
                player_health -= 10
            elif MAZE[current_pos[0]][current_pos[1]] == 3:  # Trap
                trap_effect.play()
                player_health -= 20

            # Check for health depletion
            if player_health <= 0:
                print("Game Over! The agent has died due to fire or traps.")
                gameover_effect.play()  # Play game over sound
                display_game_over("GAME OVER!")
                pygame.time.delay(2000)  # Allow sound to play
                restart_game()
            
            if agent_index < len(solution_path) - 1:
                next_pos = solution_path[agent_index + 1]

                # Check if the adversary is too close
                if calculate_distance(next_pos, adversary_path[adversary_index]) <= 3:
                    # Recalculate the agent's path to avoid adversary
                    new_path = agent_avoidance_path(solution_path[agent_index], adversary_path[adversary_index], END_POS)
                    if new_path:  # Only update if a valid path is found
                        solution_path = new_path
                        agent_index = 0  # Start over with the new path

                if adversary_index < len(adversary_path) and adversary_path[adversary_index] == next_pos:
                    print("Game Over! The adversary blocked the agent.")
                    gameover_effect.play()  # Play game over sound
                    display_game_over("GAME OVER!")
                    pygame.time.delay(2000)  # Allow sound to play
                    restart_game()

                agent_index += 1
        else:
            stun_counter -= 1  # Decrease stun counter
            if stun_counter == 0:
                agent_stunned = False

        # Adversary movement logic
        if adversary_index < len(adversary_path) - 1:
            adversary_index += 1

        if solution_path[agent_index] == adversary_path[adversary_index]:
            print("Game Over! The agent has been captured by the adversary.")
            gameover_effect.play()  # Play game over sound
            display_game_over("GAME OVER!")
            pygame.time.delay(2000)  # Allow sound to play
            restart_game()

        clock.tick(5)  # Control game speed

if __name__ == "__main__":
    main()
