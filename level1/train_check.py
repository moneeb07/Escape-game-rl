import torch
import pygame
from final import BulletDodgeEnv, DQN  # Assuming your main game file is named 'final.py'
import time

def run_game(model_path="best_model.pth", num_episodes=5):
    # Initialize environment with rendering
    env = BulletDodgeEnv(render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pygame.init()
    
    try:
        # Initialize model with correct input dimensions (22 for your environment)
        model = DQN(22, env.action_space.n).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        
        print(f"\nLoaded model from {model_path}")
        print(f"Best reward achieved during training: {checkpoint.get('best_reward', 'N/A')}")
        print("\nGame Controls:")
        print("- SPACE: Start next episode")
        print("- Q: Quit game")
        print("\nStarting game in 3 seconds...")
        time.sleep(3)
        
        episode = 0
        clock = pygame.time.Clock()
        running = True
        
        while running and episode < num_episodes:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            done = False
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1} starting...")
            
            # Main game loop
            while not done and steps < 2000:  # Maximum 2000 steps per episode
                clock.tick(60)  # 60 FPS
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                            break
                
                if not running:
                    break
                
                # Get model's action
                with torch.no_grad():
                    action = model(state).argmax().item()
                
                # Execute action
                next_state, reward, terminated, _, _ = env.step(action)
                env.render()
                
                total_reward += reward
                steps += 1
                
                # Update state
                state = torch.tensor(next_state, dtype=torch.float32).to(device)
                done = terminated
                
                # Print status updates
                if env.clue_collected:
                    print("Clue collected!")
                if reward >= 6000:  # Victory condition
                    print("Goal reached!")
            
            print(f"Episode {episode + 1} finished:")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Steps Taken: {steps}")
            
            episode += 1
            
            # Wait for space between episodes
            if running and episode < num_episodes:
                print("\nPress SPACE for next episode or Q to quit")
                waiting = True
                while waiting and running:
                    clock.tick(60)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            elif event.key == pygame.K_q:
                                running = False
                                waiting = False
    
    except FileNotFoundError:
        print(f"Error: Could not find model file '{model_path}'")
        print("Please ensure the model file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    run_game(model_path="checkpoint_episode_200.pth", num_episodes=5)