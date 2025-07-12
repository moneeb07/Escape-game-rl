import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from gymnasium import spaces

class BulletDodgeEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.screen_width = 1200
        self.screen_height = 800
        
        # Entity properties
        self.player_size = 64
        self.bullet_size = 16
        self.ghost_size = 32
        self.player_velocity = 7
        self.bullet_velocity = 4.5
        self.ghost_velocity = 2
        self.clue_size = 25
        self.escape_gate_size = 50
        
        # Health system
        self.max_health = 5
        self.current_health = self.max_health
        
        # Saw entity properties
        self.saw_size = 32
        self.saw_velocity = 3
        self.saw_direction = 1
        
        # Goal states
        self.clue_collected = False
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # No Action, Up, Down, Left, Right
        
        # Update observation space to match the actual observation dimension (19 features)
        self.observation_space = spaces.Box(
            low=0,
            high=1,  # Since we normalize all values
            shape=(19,),  # Updated to match the observation vector length
            dtype=np.float32
        )
        
        
        
        
        
        
        
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Enhanced Bullet Dodge RL")
        
        # Load images
        try:
            self.player_img = pygame.image.load('player.png')
            self.player_img = pygame.transform.scale(self.player_img, (self.player_size, self.player_size))
            
            self.bullet_img = pygame.image.load('bullet.png')
            self.ghost_img = pygame.image.load('ghost.png')
            self.saw_img = pygame.image.load('saw.png')
            self.saw_img = pygame.transform.scale(self.saw_img, (self.saw_size, self.saw_size))
            
            self.clue_img = pygame.image.load('clue.png')
            self.clue_img = pygame.transform.scale(self.clue_img, (self.clue_size, self.clue_size))
            
            self.gate_open_img = pygame.image.load('gate_open.png')
            self.gate_open_img = pygame.transform.scale(self.gate_open_img, (self.escape_gate_size, self.escape_gate_size))
            
            self.gate_close_img = pygame.image.load('gate_close.png')
            self.gate_close_img = pygame.transform.scale(self.gate_close_img, (self.escape_gate_size, self.escape_gate_size))
            
            
            self.background_img = pygame.image.load('background.png')
            self.background_img = pygame.transform.scale(self.background_img, (self.screen_width, self.screen_height))
            
        except:
            # Fallback to colored rectangles if images fail to load
            self.player_img = pygame.Surface((self.player_size, self.player_size))
            self.player_img.fill((255, 0, 0))
            self.bullet_img = pygame.Surface((self.bullet_size, self.bullet_size))
            self.bullet_img.fill((0, 0, 255))
            self.ghost_img = pygame.Surface((self.ghost_size, self.ghost_size))
            self.ghost_img.fill((128, 0, 128))
            self.saw_img = pygame.Surface((self.saw_size, self.saw_size))
            self.saw_img.fill((255, 165, 0))
            self.clue_img = pygame.Surface((50, 50))
            self.clue_img.fill((0, 255, 0))
            
            self.gate_open_img = pygame.Surface((self.escape_gate_size, self.escape_gate_size))
            self.gate_open_img.fill((0, 255, 0))
            
            self.gate_close_img = pygame.Surface((self.escape_gate_size, self.escape_gate_size))
            self.gate_close_img.fill((255, 0, 0))
        
        self.render_mode = render_mode
        self.reset()

    def _generate_goals(self):
        # Generate clue position in the right third of the screen
        min_x = int(0.7 * self.screen_width)
        self.clue_x = random.randint(min_x, self.screen_width - 50)
        self.clue_y = random.randint(0, self.screen_height - 50)
        
        # Generate escape gate position on the left side
        self.gate_x = 0
        self.gate_y = random.randint(0, self.screen_height // 3)
        
        
    def _move_saw(self):
        """Move saw up and down in the center of screen"""
        self.saw_y += self.saw_velocity * self.saw_direction
        
        # Reverse direction at screen boundaries
        if self.saw_y >= self.screen_height - self.saw_size:
            self.saw_direction = -1
        elif self.saw_y <= 0:
            self.saw_direction = 1    
    

    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random spawn in the left-bottom quadrant
        self.player_x = random.randint(0, self.screen_width // 2 - self.player_size)
        self.player_y = random.randint(self.screen_height // 2, self.screen_height - self.player_size)
        
        # Reset health and state
        self.current_health = self.max_health
        self.clue_collected = False
        
        # Reset bullet
        self.bullet_x = self.screen_width // 2
        self.bullet_y = 0
        self.bullet_dx = 0  # Initialize bullet direction
        self.bullet_dy = self.bullet_velocity
        
        # Reset ghost
        self.ghost_x = random.randint(self.screen_width // 2, self.screen_width - self.ghost_size)
        self.ghost_y = random.randint(0, self.screen_height - self.ghost_size)
        
        # Reset saw
        self.saw_x = self.screen_width // 2 - self.saw_size // 2
        self.saw_y = 0
        self.saw_direction = 1
        
        # Generate goals
        self._generate_goals()
        
        self.prev_player_pos = (self.player_x, self.player_y)
        self._predict_bullet_trajectory()
        self.steps_since_last_progress = 0
        
        if hasattr(self, 'last_action'):
            del self.last_action
            
        observation = np.array([
            self.player_x / self.screen_width,
            self.player_y / self.screen_height,
            self.bullet_x / self.screen_width,
            self.bullet_y / self.screen_height,
            self.ghost_x / self.screen_width,
            self.ghost_y / self.screen_height,
            self.saw_x / self.screen_width,
            self.saw_y / self.screen_height,
            self.clue_x / self.screen_width,
            self.clue_y / self.screen_height,
            self.gate_x / self.screen_width,
            self.gate_y / self.screen_height,
            self.bullet_dx / self.bullet_velocity,
            self.bullet_dy / self.bullet_velocity,
            self.current_health / self.max_health,
            float(self.clue_collected),
            # Calculate distances for the initial state
            np.sqrt((self.player_x - self.clue_x) ** 2 + (self.player_y - self.clue_y) ** 2) / self.screen_width,
            np.sqrt((self.player_x - self.gate_x) ** 2 + (self.player_y - self.gate_y) ** 2) / self.screen_width,
            np.sqrt((self.player_x - self.saw_x) ** 2 + (self.player_y - self.saw_y) ** 2) / self.screen_width
        ], dtype=np.float32)
        
        return observation, {}
    
    
    
    
    

    def _predict_bullet_trajectory(self):
        dx = self.player_x - self.prev_player_pos[0]
        dy = self.player_y - self.prev_player_pos[1]
        
        prediction_future_dist = 5
        predicted_player_x = self.player_x + dx * prediction_future_dist
        predicted_player_y = self.player_y + dy * prediction_future_dist
        
        predicted_player_x = max(0, min(predicted_player_x, self.screen_width - self.player_size))
        predicted_player_y = max(0, min(predicted_player_y, self.screen_height - self.player_size))
        
        dx_to_target = predicted_player_x - self.bullet_x
        dy_to_target = predicted_player_y - self.bullet_y
        
        distance_to_target = math.sqrt(dx_to_target ** 2 + dy_to_target ** 2)
        
        if distance_to_target > 0:
            self.bullet_dx = (dx_to_target / distance_to_target) * self.bullet_velocity
            self.bullet_dy = (dy_to_target / distance_to_target) * self.bullet_velocity
        else:
            self.bullet_dx = 0
            self.bullet_dy = self.bullet_velocity

    def _move_ghost(self):
        dx = self.player_x - self.ghost_x
        dy = self.player_y - self.ghost_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        if distance > 0:
            self.ghost_x += (dx / distance) * self.ghost_velocity
            self.ghost_y += (dy / distance) * self.ghost_velocity

    
    
    
    
    
    
    
    
    
    
    
    
    
    def step(self, action):
        self.prev_player_pos = (self.player_x, self.player_y)
        
        # Calculate initial distances
        initial_clue_distance = np.sqrt((self.player_x - self.clue_x) ** 2 + (self.player_y - self.clue_y) ** 2)
        initial_gate_distance = np.sqrt((self.player_x - self.gate_x) ** 2 + (self.player_y - self.gate_y) ** 2)

        # Save previous distances to threats
        prev_bullet_dist = np.sqrt((self.player_x - self.bullet_x) ** 2 + (self.player_y - self.bullet_y) ** 2)
        prev_ghost_dist = np.sqrt((self.player_x - self.ghost_x) ** 2 + (self.player_y - self.ghost_y) ** 2)
        prev_saw_dist = np.sqrt((self.player_x - self.saw_x) ** 2 + (self.player_y - self.saw_y) ** 2)

        # Move player
        if action == 1:  # Up
            self.player_y = max(0, self.player_y - self.player_velocity)
        elif action == 2:  # Down
            self.player_y = min(self.screen_height - self.player_size, self.player_y + self.player_velocity)
        elif action == 3:  # Left
            self.player_x = max(0, self.player_x - self.player_velocity)
        elif action == 4:  # Right
            self.player_x = min(self.screen_width - self.player_size, self.player_x + self.player_velocity)

        # Move entities
        self.bullet_x += self.bullet_dx
        self.bullet_y += self.bullet_dy
        self._move_ghost()
        self._move_saw()

        # Define collision rectangles
        player_rect = pygame.Rect(self.player_x, self.player_y, self.player_size, self.player_size)
        bullet_rect = pygame.Rect(self.bullet_x, self.bullet_y, self.bullet_size, self.bullet_size)
        ghost_rect = pygame.Rect(self.ghost_x, self.ghost_y, self.ghost_size, self.ghost_size)
        saw_rect = pygame.Rect(self.saw_x, self.saw_y, self.saw_size, self.saw_size)
        clue_rect = pygame.Rect(self.clue_x, self.clue_y, self.clue_size, self.clue_size)
        gate_rect = pygame.Rect(self.gate_x, self.gate_y, self.escape_gate_size, self.escape_gate_size)

        # Calculate new distances
        new_bullet_dist = np.sqrt((self.player_x - self.bullet_x) ** 2 + (self.player_y - self.bullet_y) ** 2)
        new_ghost_dist = np.sqrt((self.player_x - self.ghost_x) ** 2 + (self.player_y - self.ghost_y) ** 2)
        new_saw_dist = np.sqrt((self.player_x - self.saw_x) ** 2 + (self.player_y - self.saw_y) ** 2)

        reward = 0
        terminated = False

        # Defensive rewards
        for prev_dist, new_dist, safe_dist, penalty, bonus in [
            (prev_bullet_dist, new_bullet_dist, 150, -500, 20),
            (prev_ghost_dist, new_ghost_dist, 120, -600, 25),
            (prev_saw_dist, new_saw_dist, 120, -600, 25)
        ]:
            if new_dist < safe_dist:
                # Penalty for getting closer
                reward += penalty * (safe_dist - new_dist) / safe_dist
            else:
                # Bonus for maintaining safe distance
                reward += bonus * (1 - safe_dist / new_dist)

        # Collision penalties
        if player_rect.colliderect(bullet_rect):
            reward -= 1000
            self.current_health -= 1
            self.bullet_x, self.bullet_y = self.screen_width // 2, 0
            self._predict_bullet_trajectory()

        if player_rect.colliderect(ghost_rect):
            reward -= 1200
            self.current_health -= 1
            self.ghost_x = random.randint(self.screen_width // 2, self.screen_width - self.ghost_size)
            self.ghost_y = random.randint(0, self.screen_height - self.ghost_size)

        if player_rect.colliderect(saw_rect):
            reward -= 1200
            self.current_health -= 1
            self.player_x = max(0, self.player_x - 40 if self.player_x < self.saw_x else self.player_x + 40)

        # Goal rewards
        if not self.clue_collected:
            clue_progress = initial_clue_distance - np.sqrt((self.player_x - self.clue_x) ** 2 + (self.player_y - self.clue_y) ** 2)
            reward += clue_progress * 20
            if player_rect.colliderect(clue_rect):
                self.clue_collected = True
                reward += 5000
        else:
            gate_progress = initial_gate_distance - np.sqrt((self.player_x - self.gate_x) ** 2 + (self.player_y - self.gate_y) ** 2)
            reward += gate_progress * 25
            if player_rect.colliderect(gate_rect):
                reward += 10000
                terminated = True

        # Health penalty for running out of health
        if self.current_health <= 0:
            reward -= 1500
            terminated = True

        # Anti-stagnation penalty
        self.steps_since_last_progress += 1
        if self.steps_since_last_progress > 40:
            reward -= 50
            self.steps_since_last_progress = 0

        # Reset off-screen bullet
        if self.bullet_y > self.screen_height or self.bullet_x < 0 or self.bullet_x > self.screen_width:
            self.bullet_x, self.bullet_y = self.screen_width // 2, 0
            self._predict_bullet_trajectory()

        return self._get_obs(), reward, terminated, False, {}

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _get_obs(self):
        # Enhanced observation space with normalized distances and goal indicators
        normalized_x = self.player_x / self.screen_width
        normalized_y = self.player_y / self.screen_height
        
        # Calculate normalized distances to objectives
        dist_to_clue = np.sqrt((self.player_x - self.clue_x) ** 2 + 
                            (self.player_y - self.clue_y) ** 2) / self.screen_width
        dist_to_gate = np.sqrt((self.player_x - self.gate_x) ** 2 + 
                            (self.player_y - self.gate_y) ** 2) / self.screen_width
        
        dist_to_saw = np.sqrt((self.player_x - self.saw_x) ** 2 + 
                             (self.player_y - self.saw_y) ** 2) / self.screen_width
        
        return np.array([
            normalized_x,
            normalized_y,
            self.bullet_x / self.screen_width,
            self.bullet_y / self.screen_height,
            self.ghost_x / self.screen_width,
            self.ghost_y / self.screen_height,
            self.saw_x / self.screen_width,
            self.saw_y / self.screen_height,
            self.clue_x / self.screen_width,
            self.clue_y / self.screen_height,
            self.gate_x / self.screen_width,
            self.gate_y / self.screen_height,
            self.bullet_dx / self.bullet_velocity,
            self.bullet_dy / self.bullet_velocity,
            self.current_health / self.max_health,
            float(self.clue_collected),
            dist_to_clue,
            dist_to_gate,
            dist_to_saw
        ], dtype=np.float32)
    
    
    def render(self):
        if self.render_mode == 'human':
            self.screen.fill((0, 200, 150))
            self.screen.blit(self.background_img, (0, 0))
            
            # Draw health bar
            health_width = 200
            health_height = 20
            health_x = 10
            health_y = 10
            pygame.draw.rect(self.screen, (255, 0, 0), 
                           (health_x, health_y, health_width, health_height))
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (health_x, health_y, 
                            health_width * (self.current_health / self.max_health),
                            health_height))
            
            # Draw clue 
            self.screen.blit(self.clue_img, (self.clue_x, self.clue_y))
            
            
            # Only draw gate if clue is collected
            if self.clue_collected:
                self.screen.blit(self.gate_open_img, (self.gate_x, self.gate_y))
            else:
                self.screen.blit(self.gate_close_img, (self.gate_x, self.gate_y))
            
          
            
            # Draw entities
            self.screen.blit(self.player_img, (self.player_x, self.player_y))
            self.screen.blit(self.bullet_img, (self.bullet_x, self.bullet_y))
            self.screen.blit(self.ghost_img, (self.ghost_x, self.ghost_y))
            self.screen.blit(self.saw_img, (self.saw_x, self.saw_y))
            
            pygame.display.flip()

    def close(self):
        pygame.quit()



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Define layers with layer normalization
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        
        self.fc5 = nn.Linear(128, output_dim)
        
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Add batch dimension if input is a single sample
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.fc5(x)
        
        # Remove batch dimension if input was a single sample
        if x.size(0) == 1:
            x = x.squeeze(0)
            
        return x



def train_dqn(env, episodes=2000, learning_rate=5e-4):  # Increased episodes, adjusted learning rate
    clock = pygame.time.Clock()
    render_freq = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(19, env.action_space.n).to(device)
    target_net = DQN(19, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=50
    )
    criterion = nn.HuberLoss()  # Changed to Huber loss for better stability
    
    replay_buffer = []
    batch_size = 512  # Increased batch size
    gamma = 0.995  # Increased discount factor
    epsilon = 1.0
    epsilon_min = 0.02  # Lower minimum epsilon
    epsilon_decay = 0.998
    max_buffer_size = 100000  # Increased buffer size
    target_update_freq = 5
    
    # Priority weights for sampling
    priority_alpha = 0.6
    priority_beta = 0.4
    priority_beta_increment = 0.001

    best_total_reward = float('-inf')
    no_improvement_count = 0
    patience = 100  # Episodes to wait before early stopping

    def print_lr():
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False
        total_reward = 0
        steps = 0
        prev_action = None
        action_repeat_count = 0

        while not done:
            env.render()     # temporarily removed 
            
            
            
            
            
            
            steps += 1

            # Decay epsilon with a minimum threshold
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Action selection with action repeat penalty
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    # Penalize repeated actions when close to goals
                    if prev_action is not None and action_repeat_count > 3:
                        q_values[prev_action] *= 0.8
                    action = q_values.argmax().item()

            # Track action repetition
            if action == prev_action:
                action_repeat_count += 1
            else:
                action_repeat_count = 0
            prev_action = action

            next_state, reward, terminated, _, _ = env.step(action)
            done = terminated
            total_reward += reward

            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            
            # Add experience with priority
            td_error = abs(reward + gamma * policy_net(next_state).max().item() - 
                         policy_net(state).gather(0, torch.tensor([action]).to(device)).item())
            priority = (td_error + 1e-6) ** priority_alpha
            
            replay_buffer.append((state, action, reward, next_state, done, priority))

            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state

            if len(replay_buffer) >= batch_size:
                # Sample batch based on priorities
                priorities = np.array([exp[5] for exp in replay_buffer])
                probs = priorities / priorities.sum()
                indices = np.random.choice(len(replay_buffer), batch_size, p=probs)
                
                batch = [replay_buffer[idx] for idx in indices]
                states, actions, rewards, next_states, dones, _ = zip(*batch)

                # Calculate importance sampling weights
                weights = (len(replay_buffer) * probs[indices]) ** (-priority_beta)
                weights = weights / weights.max()
                weights = torch.tensor(weights, dtype=torch.float32).to(device)

                states = torch.stack(states)
                next_states = torch.stack(next_states)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = target_net(next_states).max(1)[0]
                target_q = rewards + gamma * next_q * (1 - dones)

                # Apply importance sampling weights to loss
                loss = (weights * criterion(current_q, target_q.detach())).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1)
                optimizer.step()

                # Update priority beta
                priority_beta = min(1.0, priority_beta + priority_beta_increment)

            if steps > 2000:  # Increased max steps
                break

        # Update learning rate based on performance
        scheduler.step(total_reward)
        # Save checkpoints every 100 episodes
        if episode % 100 == 0:
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_reward': best_total_reward,
                'replay_buffer': replay_buffer[-1000:],  # Save last 1000 experiences
                'total_steps': steps
            }, f'checkpoint_episode_{episode}.pth')

        # Early stopping check
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            no_improvement_count = 0
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_reward': best_total_reward,
                'total_steps': steps
            }, "best_model.pth")
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {episode + 1} episodes")
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'best_reward': best_total_reward,
                'replay_buffer': replay_buffer[-1000:],  # Save last 1000 experiences
                'total_steps': steps
            }, f'last_iter_{episode}.pth')
            break

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # if episode % 10 == 0:
        print(f"Ep {episode + 1}, TR: {total_reward:.2f}, Epsilon: {epsilon:.2f}, Steps: {steps}")
        print_lr()

    return policy_net




def evaluate_model(env, policy_net, num_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_rewards = []
    success_count = 0
    clue_success_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False
        total_reward = 0
        reached_goal = False
        reached_clue = False

        while not done:
            env.render()

            with torch.no_grad():
                action = policy_net(state).argmax().item()

            next_state, reward, terminated, _, _ = env.step(action)
            done = terminated
            total_reward += reward

            # Check if clue was collected this step
            if env.clue_collected and not reached_clue:
                reached_clue = True
                clue_success_count += 1

            # Check if final goal was reached
            if reward >= 300:  # Final goal reward
                reached_goal = True
                success_count += 1

            state = torch.tensor(next_state, dtype=torch.float32).to(device)

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward:.2f}, " +
              f"Reached Clue = {reached_clue}, Reached Final Goal = {reached_goal}")

    success_rate = (success_count / num_episodes) * 100
    clue_success_rate = (clue_success_count / num_episodes) * 100
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Clue Collection Rate: {clue_success_rate:.2f}%")
    print(f"Final Goal Success Rate: {success_rate:.2f}%")
    
    # Save evaluation results
    torch.save({
        'total_rewards': total_rewards,
        'success_rate': success_rate,
        'clue_success_rate': clue_success_rate,
        'average_reward': np.mean(total_rewards)
    }, "evaluation_results.pth")
    
    return np.mean(total_rewards)

def main():
    # env = BulletDodgeEnv(render_mode='human')
    env = BulletDodgeEnv(render_mode='Human')

    try:
        print("Starting Training...")
        policy_net = train_dqn(env, episodes=1000)

        print("\nStarting Evaluation...")
        evaluate_model(env, policy_net)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()