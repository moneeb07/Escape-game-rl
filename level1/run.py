import torch
import pygame
from final import BulletDodgeEnv, DQN  # Assuming your main game file is named 'final.py'

def run_level1(model_path="best_model.pth"):
    # Initialize environment
    env = BulletDodgeEnv(render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize model
        model = DQN(22, env.action_space.n).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        
        # Run a single episode
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False
        goal_reached = False
        
        
        
        clock = pygame.time.Clock()
        while not done:
            clock.tick(30)
            # Get model's action
            with torch.no_grad():
                action = model(state).argmax().item()
            
            # Execute action
            next_state, reward, terminated, _, _ = env.step(action)
            env.render()
            
            # Check if goal is reached (assuming reward >= 6000 indicates goal reached)
            if reward >= 6000:
                goal_reached = True
                done = True
            
            # Update state
            state = torch.tensor(next_state, dtype=torch.float32).to(device)
            done = terminated or done
            
            # Handle Pygame events to allow window closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        return goal_reached

    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    result = run_level1()
    print(f"Goal reached: {result}")