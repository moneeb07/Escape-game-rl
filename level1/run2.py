import torch
import pygame
from final2 import BulletDodgeEnv, DQN

def run_game_once(model_path="2last_iter_105.pth"):
    # Initialize environment
    env = BulletDodgeEnv(render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize model
        model = DQN(19, env.action_space.n).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['policy_net_state_dict'])
        model.eval()
        
        # Run a single episode
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False
        gate_reached = False
        clock = pygame.time.Clock()
        while not done:
            # Get model's action
            clock.tick(30)
            with torch.no_grad():
                action = model(state).argmax().item()
            
            # Execute action
            next_state, reward, terminated, _, _ = env.step(action)
            env.render()
            
            # Check if gate is reached (assuming reward >= 10000 indicates gate reached)
            if reward >= 10000:
                gate_reached = True
                done = True
            
            # Update state
            state = torch.tensor(next_state, dtype=torch.float32).to(device)
            done = terminated or done
            
            # Handle Pygame events to allow window closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        return gate_reached

    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    result = run_game_once()
    print(f"Gate reached: {result}")