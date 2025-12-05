
"""
PPO CartPole - Beautiful Arcade Game UI with Live Graphics
Real-time animated game with proper graphics and image capture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import pickle
import os
import hashlib
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not installed. Install with: pip install pygame")


# ============================================================================
# SECURITY_LAYER - Basic security
# ============================================================================
class SECURITY_LAYER:
    """Security layer for models"""
    
    @staticmethod
    def HASH_FILE(filepath):
        """Generate file hash"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def VERIFY_MODEL_INTEGRITY(model_file, hash_file):
        """Verify model integrity"""
        try:
            if not os.path.exists(hash_file):
                return False
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            current_hash = SECURITY_LAYER.HASH_FILE(model_file)
            return stored_hash == current_hash
        except:
            return False
    
    @staticmethod
    def SAVE_MODEL_SECURE(agent, filename, description=""):
        """Save model securely"""
        try:
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            model_file = model_dir / f"{filename}.pkl"
            hash_file = model_dir / f"{filename}_hash.txt"
            metadata_file = model_dir / f"{filename}_metadata.json"
            
            model_data = {
                'actor_state': agent.actor.state_dict(),
                'critic_state': agent.critic.state_dict(),
                'state_size': agent.state_size,
                'action_size': agent.action_size
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            file_hash = SECURITY_LAYER.HASH_FILE(model_file)
            with open(hash_file, 'w') as f:
                f.write(file_hash)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'description': description,
                'hash': file_hash,
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Model saved: {model_file}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    @staticmethod
    def LOAD_MODEL_SECURE(agent, filename):
        """Load model securely"""
        try:
            model_dir = Path("models")
            model_file = model_dir / f"{filename}.pkl"
            hash_file = model_dir / f"{filename}_hash.txt"
            
            if not model_file.exists():
                return False
            
            if not SECURITY_LAYER.VERIFY_MODEL_INTEGRITY(model_file, hash_file):
                return False
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            agent.actor.load_state_dict(model_data['actor_state'])
            agent.critic.load_state_dict(model_data['critic_state'])
            
            print(f"✓ Model loaded: {model_file}")
            return True
        except:
            return False


def SETUP_SYSTEM():
    """Setup system"""
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    return device


class ACTOR_NETWORK(nn.Module):
    """Actor Network"""
    def __init__(self, state_size, action_size, learning_rate=1e-4):
        super(ACTOR_NETWORK, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)


class CRITIC_NETWORK(nn.Module):
    """Critic Network"""
    def __init__(self, state_size, learning_rate=1e-4):
        super(CRITIC_NETWORK, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPO_AGENT:
    """PPO Agent"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = ACTOR_NETWORK(state_size, action_size)
        self.critic = CRITIC_NETWORK(state_size)
        self.memory = {
            'states': [], 'actions': [], 'rewards': [], 
            'values': [], 'log_probs': [], 'dones': []
        }
    
    def SELECT_ACTION(self, state):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        action_probs = action_probs.cpu().numpy()[0]
        action = np.random.choice(self.action_size, p=action_probs)
        log_prob = np.log(action_probs[action] + 1e-8)
        return action, value.cpu().item(), log_prob
    
    def REMEMBER_EXPERIENCE(self, state, action, reward, value, log_prob, done):
        """Remember experience"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def COMPUTE_ADVANTAGES(self, next_value=0):
        """Compute advantages"""
        advantages = []
        gae = 0
        values = self.memory['values'] + [next_value]
        
        for t in reversed(range(len(self.memory['rewards']))):
            if self.memory['dones'][t]:
                gae = 0
            else:
                delta = self.memory['rewards'][t] + 0.99 * values[t + 1] * (1 - self.memory['dones'][t]) - values[t]
                gae = delta + 0.99 * 0.95 * (1 - self.memory['dones'][t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(self.memory['values'])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns
    
    def UPDATE_NETWORKS(self, epochs=10):
        """Update networks"""
        if len(self.memory['states']) == 0:
            return
        
        states = np.array(self.memory['states'])
        actions = np.array(self.memory['actions'])
        old_log_probs = np.array(self.memory['log_probs'])
        
        last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.critic(last_state).cpu().item()
        
        advantages, returns = self.COMPUTE_ADVANTAGES(next_value)
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        for epoch in range(epochs):
            action_probs = self.actor(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_tensor
            
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist.entropy().mean()
            
            self.actor.optimizer.zero_grad()
            (actor_loss + 0.01 * entropy_loss).backward()
            self.actor.optimizer.step()
            
            values = self.critic(states_tensor).squeeze()
            critic_loss = nn.MSELoss()(values, returns_tensor)
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
        
        self.memory = {
            'states': [], 'actions': [], 'rewards': [], 
            'values': [], 'log_probs': [], 'dones': []
        }


def TRAINING_LOOP(episodes=150):
    """Train agent"""
    print("="*70)
    print("TRAINING AGENT")
    print("="*70)
    
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = PPO_AGENT(state_size, action_size)
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 500:
            action, value, log_prob = agent.SELECT_ACTION(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.REMEMBER_EXPERIENCE(state, action, reward, value, log_prob, done)
            state = next_state
            episode_reward += reward
            step += 1
        
        agent.UPDATE_NETWORKS(epochs=5)
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 30 == 0:
            avg = np.mean(episode_rewards[-30:])
            print(f"Episode {episode + 1:4d}/{episodes} | Avg: {avg:7.2f} | Current: {episode_reward:7.2f}")
    
    env.close()
    print("="*70 + "\n")
    return agent, episode_rewards


# ============================================================================
# DRAW_GAME - Beautiful arcade-style game graphics
# ============================================================================
def DRAW_ARCADE_GAME(screen, WIDTH, HEIGHT, game_num, step, max_steps, score1, score2, state1, state2):
    """Draw arcade-style game graphics"""
    
    # Background color (arcade brown)
    screen.fill((180, 120, 80))
    
    # Top header bar
    pygame.draw.rect(screen, (230, 230, 230), (0, 0, WIDTH, 60))
    pygame.draw.rect(screen, (50, 50, 50), (0, 0, WIDTH, 60), 3)
    
    # Game title
    font_title = pygame.font.Font(None, 44)
    title = font_title.render("PPO CARTPOLE BATTLE", True, (255, 100, 0))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 12))
    
    # Game info
    font_small = pygame.font.Font(None, 24)
    game_info = font_small.render(f"Game {game_num} | Step {step}/{max_steps}", True, (50, 50, 50))
    screen.blit(game_info, (20, 20))
    
    # Divider line in middle
    pygame.draw.line(screen, (100, 100, 100), (WIDTH//2, 60), (WIDTH//2, HEIGHT), 4)
    
    # ========== PLAYER 1 (LEFT) ==========
    LEFT_X = 50
    PLAY_WIDTH = WIDTH//2 - 70
    PLAY_HEIGHT = HEIGHT - 80
    GROUND_Y = 300
    
    # Player 1 background
    pygame.draw.rect(screen, (140, 100, 60), (LEFT_X, 80, PLAY_WIDTH, PLAY_HEIGHT))
    pygame.draw.rect(screen, (200, 150, 80), (LEFT_X, 80, PLAY_WIDTH, PLAY_HEIGHT), 3)
    
    # Player 1 label
    font_large = pygame.font.Font(None, 36)
    p1_label = font_large.render("PLAYER 1", True, (100, 150, 255))
    screen.blit(p1_label, (LEFT_X + 20, 90))
    
    # Draw Player 1 game area
    cart_x = int(LEFT_X + 50 + state1[0] * 40)
    
    # Ground
    pygame.draw.rect(screen, (100, 80, 40), (LEFT_X + 20, GROUND_Y + 80, PLAY_WIDTH - 40, 30))
    pygame.draw.line(screen, (50, 50, 50), (LEFT_X + 20, GROUND_Y + 80), (LEFT_X + PLAY_WIDTH - 20, GROUND_Y + 80), 3)
    
    # Cart (blue square)
    CART_SIZE = 40
    pygame.draw.rect(screen, (100, 150, 255), (cart_x - CART_SIZE//2, GROUND_Y + 50, CART_SIZE, CART_SIZE))
    pygame.draw.rect(screen, (50, 100, 200), (cart_x - CART_SIZE//2, GROUND_Y + 50, CART_SIZE, CART_SIZE), 2)
    
    # Wheels
    pygame.draw.circle(screen, (50, 50, 50), (int(cart_x - 12), GROUND_Y + 90), 5)
    pygame.draw.circle(screen, (50, 50, 50), (int(cart_x + 12), GROUND_Y + 90), 5)
    
    # Pole (stick from cart)
    pole_angle = state1[2]
    pole_length = 80
    pole_end_x = int(cart_x + pole_length * np.sin(pole_angle))
    pole_end_y = int(GROUND_Y + 50 - pole_length * np.cos(pole_angle))
    pygame.draw.line(screen, (255, 200, 100), (int(cart_x), GROUND_Y + 50), (pole_end_x, pole_end_y), 5)
    pygame.draw.circle(screen, (255, 255, 0), (pole_end_x, pole_end_y), 8)
    
    # Score display
    font_score = pygame.font.Font(None, 48)
    score_text = font_score.render(f"SCORE: {int(score1)}", True, (100, 150, 255))
    screen.blit(score_text, (LEFT_X + 20, GROUND_Y + 150))
    
    # State info
    state_info = font_small.render(f"Pos: {state1[0]:6.2f} | Vel: {state1[1]:6.2f}", True, (255, 255, 255))
    screen.blit(state_info, (LEFT_X + 20, GROUND_Y + 220))
    
    # ========== PLAYER 2 (RIGHT) ==========
    RIGHT_X = WIDTH//2 + 20
    
    # Player 2 background
    pygame.draw.rect(screen, (140, 100, 60), (RIGHT_X, 80, PLAY_WIDTH, PLAY_HEIGHT))
    pygame.draw.rect(screen, (200, 100, 80), (RIGHT_X, 80, PLAY_WIDTH, PLAY_HEIGHT), 3)
    
    # Player 2 label
    p2_label = font_large.render("PLAYER 2", True, (255, 100, 100))
    screen.blit(p2_label, (RIGHT_X + 20, 90))
    
    # Draw Player 2 game area
    cart_x = int(RIGHT_X + 50 + state2[0] * 40)
    
    # Ground
    pygame.draw.rect(screen, (100, 80, 40), (RIGHT_X + 20, GROUND_Y + 80, PLAY_WIDTH - 40, 30))
    pygame.draw.line(screen, (50, 50, 50), (RIGHT_X + 20, GROUND_Y + 80), (RIGHT_X + PLAY_WIDTH - 20, GROUND_Y + 80), 3)
    
    # Cart (red square)
    pygame.draw.rect(screen, (255, 100, 100), (cart_x - CART_SIZE//2, GROUND_Y + 50, CART_SIZE, CART_SIZE))
    pygame.draw.rect(screen, (200, 50, 50), (cart_x - CART_SIZE//2, GROUND_Y + 50, CART_SIZE, CART_SIZE), 2)
    
    # Wheels
    pygame.draw.circle(screen, (50, 50, 50), (int(cart_x - 12), GROUND_Y + 90), 5)
    pygame.draw.circle(screen, (50, 50, 50), (int(cart_x + 12), GROUND_Y + 90), 5)
    
    # Pole
    pole_angle = state2[2]
    pole_end_x = int(cart_x + pole_length * np.sin(pole_angle))
    pole_end_y = int(GROUND_Y + 50 - pole_length * np.cos(pole_angle))
    pygame.draw.line(screen, (255, 200, 100), (int(cart_x), GROUND_Y + 50), (pole_end_x, pole_end_y), 5)
    pygame.draw.circle(screen, (255, 255, 0), (pole_end_x, pole_end_y), 8)
    
    # Score display
    score_text = font_score.render(f"SCORE: {int(score2)}", True, (255, 100, 100))
    screen.blit(score_text, (RIGHT_X + 20, GROUND_Y + 150))
    
    # State info
    state_info = font_small.render(f"Pos: {state2[0]:6.2f} | Vel: {state2[1]:6.2f}", True, (255, 255, 255))
    screen.blit(state_info, (RIGHT_X + 20, GROUND_Y + 220))


# ============================================================================
# LIVE_GAME_UI - Real-time pygame game with arcade graphics
# ============================================================================
def LIVE_GAME_UI(agent1, agent2, max_steps=500, num_games=3):
    """Live game with arcade graphics"""
    
    if not PYGAME_AVAILABLE:
        print("❌ pygame required. Install: pip install pygame")
        return
    
    pygame.init()
    
    WIDTH = 1400
    HEIGHT = 800
    FPS = 60
    
    Path("game_images").mkdir(exist_ok=True)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PPO CartPole - AI Battle Arena")
    clock = pygame.time.Clock()
    
    env = gym.make('CartPole-v1')
    
    total_p1_wins = 0
    total_p2_wins = 0
    
    for game_num in range(num_games):
        state1, _ = env.reset()
        state2, _ = env.reset()
        
        score1 = 0
        score2 = 0
        step = 0
        done1 = False
        done2 = False
        
        print(f"\nGame {game_num + 1}/{num_games}")
        
        running = True
        while running and step < max_steps:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    elif event.key == pygame.K_SPACE:
                        step = max_steps
            
            # Game logic
            if not done1:
                action1, _, _ = agent1.SELECT_ACTION(state1)
                next_state1, reward1, terminated1, truncated1, _ = env.step(action1)
                done1 = terminated1 or truncated1
                if not done1:
                    state1 = next_state1
                score1 += reward1
            
            if not done2:
                action2, _, _ = agent2.SELECT_ACTION(state2)
                next_state2, reward2, terminated2, truncated2, _ = env.step(action2)
                done2 = terminated2 or truncated2
                if not done2:
                    state2 = next_state2
                score2 += reward2
            
            step += 1
            
            # Draw game
            DRAW_ARCADE_GAME(screen, WIDTH, HEIGHT, game_num + 1, step, max_steps, score1, score2, state1, state2)
            pygame.display.flip()
            
            # Capture image every 50 steps
            if step % 50 == 0 and step > 0:
                try:
                    images_dir = Path("game_images")
                    filename = images_dir / f"game_{game_num + 1}_step_{step:04d}.png"
                    pygame.image.save(screen, str(filename))
                    print(f"  📸 Screenshot: game_{game_num + 1}_step_{step:04d}.png")
                except:
                    pass
        
        # Game results
        if score1 > score2:
            total_p1_wins += 1
            print(f"✓ Game {game_num + 1}: Player 1 WINS! ({score1} vs {score2})")
        elif score2 > score1:
            total_p2_wins += 1
            print(f"✓ Game {game_num + 1}: Player 2 WINS! ({score2} vs {score1})")
        else:
            print(f"✓ Game {game_num + 1}: TIE! ({score1} vs {score2})")
    
    env.close()
    pygame.quit()
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Player 1 Wins: {total_p1_wins}")
    print(f"Player 2 Wins: {total_p2_wins}")
    print(f"{'='*70}\n")


def DISPLAY_MENU():
    """Display menu"""
    print("\n" + "*"*70)
    print("PPO CARTPOLE - ARCADE GAME BATTLE")
    print("*"*70)
    print("\n[1] Train new agents")
    print("[2] Play AI vs AI (ARCADE GAME)")
    print("[3] Save agents")
    print("[4] Load agents")
    print("[5] Exit")
    print("\n" + "-"*70)
    return input("Enter choice (1-5): ").strip()


def MAIN():
    """Main execution"""
    SETUP_SYSTEM()
    
    agent1 = None
    agent2 = None
    
    while True:
        choice = DISPLAY_MENU()
        
        if choice == '1':
            print("\nTraining Agent 1...")
            agent1, _ = TRAINING_LOOP(episodes=120)
            print("Training Agent 2...")
            agent2, _ = TRAINING_LOOP(episodes=120)
            print("\n✓ Both agents trained!")
        
        elif choice == '2':
            if agent1 is None or agent2 is None:
                print("\n❌ Both agents needed! Train first.")
            else:
                print("\n📥 Installing pygame if needed...")
                os.system("pip install pygame --quiet")
                num_games = int(input("\nHow many games? (1-5): ") or 3)
                print("📸 Images will be saved to game_images/ folder")
                LIVE_GAME_UI(agent1, agent2, num_games=min(num_games, 5))
        
        elif choice == '3':
            if agent1 and agent2:
                SECURITY_LAYER.SAVE_MODEL_SECURE(agent1, "agent1", "Agent 1")
                SECURITY_LAYER.SAVE_MODEL_SECURE(agent2, "agent2", "Agent 2")
                print("✓ Agents saved!")
        
        elif choice == '4':
            env = gym.make('CartPole-v1')
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            env.close()
            
            agent1 = PPO_AGENT(state_size, action_size)
            agent2 = PPO_AGENT(state_size, action_size)
            
            success = 0
            if SECURITY_LAYER.LOAD_MODEL_SECURE(agent1, "agent1"):
                success += 1
            if SECURITY_LAYER.LOAD_MODEL_SECURE(agent2, "agent2"):
                success += 1
            
            if success == 2:
                print("✓ Both agents loaded!")
        
        elif choice == '5':
            print("\n" + "="*70)
            print("Thank you for playing!")
            print("="*70 + "\n")
            break
        
        else:
            print("\n❌ Invalid choice!")


if __name__ == "__main__":
    try:
        MAIN()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")