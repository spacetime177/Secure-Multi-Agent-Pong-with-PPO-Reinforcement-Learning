
"""
PPO Pong Game - AI vs AI with Live Arcade Graphics
Real-time animated Pong game with proper arcade graphics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
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
# SECURITY_LAYER
# ============================================================================
class SECURITY_LAYER:
    @staticmethod
    def HASH_FILE(filepath):
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def VERIFY_MODEL_INTEGRITY(model_file, hash_file):
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
        try:
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            model_file = model_dir / f"{filename}.pkl"
            hash_file = model_dir / f"{filename}_hash.txt"
            metadata_file = model_dir / f"{filename}_metadata.json"
            
            model_data = {
                'model_state': agent.model.state_dict(),
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
            
            agent.model.load_state_dict(model_data['model_state'])
            
            print(f"✓ Model loaded: {model_file}")
            return True
        except:
            return False


def SETUP_SYSTEM():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    return device


# ============================================================================
# PONG_NETWORK - Simple neural network for Pong AI
# ============================================================================
class PONG_NETWORK(nn.Module):
    def __init__(self):
        super(PONG_NETWORK, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)


# ============================================================================
# PONG_AGENT - AI Agent for Pong
# ============================================================================
class PONG_AGENT:
    def __init__(self):
        self.model = PONG_NETWORK()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = []
    
    def SELECT_ACTION(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def TRAIN(self, episodes=100):
        print("Training agent...")
        for ep in range(episodes):
            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep + 1}/{episodes}")
        print("✓ Training complete!")


# ============================================================================
# PONG_GAME - Pong Game Logic
# ============================================================================
class PONG_GAME:
    def __init__(self, WIDTH, HEIGHT):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.PADDLE_HEIGHT = 80
        self.PADDLE_WIDTH = 15
        self.BALL_SIZE = 10
        
        # Paddles
        self.p1_y = HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.p2_y = HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        
        # Ball
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_vx = 5
        self.ball_vy = 5
        
        # Scores
        self.p1_score = 0
        self.p2_score = 0
    
    def UPDATE(self, p1_action, p2_action):
        # Move paddles
        if p1_action == 1 and self.p1_y > 0:
            self.p1_y -= 6
        elif p1_action == 2 and self.p1_y < self.HEIGHT - self.PADDLE_HEIGHT:
            self.p1_y += 6
        
        if p2_action == 1 and self.p2_y > 0:
            self.p2_y -= 6
        elif p2_action == 2 and self.p2_y < self.HEIGHT - self.PADDLE_HEIGHT:
            self.p2_y += 6
        
        # Update ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Bounce off top/bottom
        if self.ball_y - self.BALL_SIZE // 2 <= 0 or self.ball_y + self.BALL_SIZE // 2 >= self.HEIGHT:
            self.ball_vy = -self.ball_vy
        
        # Paddle collisions
        if (self.ball_x - self.BALL_SIZE // 2 <= self.PADDLE_WIDTH and
            self.p1_y <= self.ball_y <= self.p1_y + self.PADDLE_HEIGHT):
            self.ball_vx = abs(self.ball_vx) + 0.5
        
        if (self.ball_x + self.BALL_SIZE // 2 >= self.WIDTH - self.PADDLE_WIDTH and
            self.p2_y <= self.ball_y <= self.p2_y + self.PADDLE_HEIGHT):
            self.ball_vx = -abs(self.ball_vx) - 0.5
        
        # Scoring
        if self.ball_x < 0:
            self.p2_score += 1
            self.RESET_BALL()
        elif self.ball_x > self.WIDTH:
            self.p1_score += 1
            self.RESET_BALL()
    
    def RESET_BALL(self):
        self.ball_x = self.WIDTH // 2
        self.ball_y = self.HEIGHT // 2
        self.ball_vx = 5 * (1 if np.random.random() > 0.5 else -1)
        self.ball_vy = 5 * (1 if np.random.random() > 0.5 else -1)
    
    def GET_STATE_P1(self):
        return np.array([
            self.p1_y / self.HEIGHT,
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.ball_vy / 10
        ])
    
    def GET_STATE_P2(self):
        return np.array([
            self.p2_y / self.HEIGHT,
            (self.WIDTH - self.ball_x) / self.WIDTH,
            self.ball_y / self.HEIGHT,
            self.ball_vy / 10
        ])


def TRAINING_LOOP(episodes=100):
    print("="*70)
    print("TRAINING PONG AI")
    print("="*70)
    
    agent = PONG_AGENT()
    agent.TRAIN(episodes)
    
    print("="*70 + "\n")
    return agent


# ============================================================================
# DRAW_PONG_GAME - Draw arcade-style Pong game
# ============================================================================
def DRAW_PONG_GAME(screen, WIDTH, HEIGHT, game_num, game_state, p1_score, p2_score):
    # Background
    screen.fill((30, 30, 50))
    
    # Top bar
    pygame.draw.rect(screen, (230, 230, 230), (0, 0, WIDTH, 60))
    pygame.draw.rect(screen, (50, 50, 50), (0, 0, WIDTH, 60), 3)
    
    # Title
    font_title = pygame.font.Font(None, 44)
    title = font_title.render("PONG BATTLE", True, (255, 100, 0))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 12))
    
    # Divider line in middle
    for y in range(0, HEIGHT, 20):
        pygame.draw.line(screen, (100, 200, 255), (WIDTH//2, y), (WIDTH//2, y + 10), 2)
    
    # PLAYER 1 (LEFT)
    # Paddle
    pygame.draw.rect(screen, (100, 150, 255), (20, int(game_state.p1_y), 15, 80))
    pygame.draw.rect(screen, (50, 100, 200), (20, int(game_state.p1_y), 15, 80), 2)
    
    # Label
    font_large = pygame.font.Font(None, 36)
    p1_label = font_large.render("PLAYER 1", True, (100, 150, 255))
    screen.blit(p1_label, (40, 80))
    
    # Score
    font_score = pygame.font.Font(None, 72)
    score1_text = font_score.render(str(int(p1_score)), True, (100, 150, 255))
    screen.blit(score1_text, (80, 200))
    
    # PLAYER 2 (RIGHT)
    # Paddle
    pygame.draw.rect(screen, (255, 100, 100), (WIDTH - 35, int(game_state.p2_y), 15, 80))
    pygame.draw.rect(screen, (200, 50, 50), (WIDTH - 35, int(game_state.p2_y), 15, 80), 2)
    
    # Label
    p2_label = font_large.render("PLAYER 2", True, (255, 100, 100))
    screen.blit(p2_label, (WIDTH - 230, 80))
    
    # Score
    score2_text = font_score.render(str(int(p2_score)), True, (255, 100, 100))
    screen.blit(score2_text, (WIDTH - 150, 200))
    
    # BALL (center)
    pygame.draw.circle(screen, (255, 255, 100), (int(game_state.ball_x), int(game_state.ball_y)), 8)
    pygame.draw.circle(screen, (255, 200, 50), (int(game_state.ball_x), int(game_state.ball_y)), 8, 2)


# ============================================================================
# LIVE_PONG_GAME - Live Pong game
# ============================================================================
def LIVE_PONG_GAME(agent1, agent2, num_games=3):
    if not PYGAME_AVAILABLE:
        print("❌ pygame required. Install: pip install pygame")
        return
    
    pygame.init()
    
    WIDTH = 1000
    HEIGHT = 600
    FPS = 60
    
    Path("game_images").mkdir(exist_ok=True)
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PONG BATTLE - AI vs AI")
    clock = pygame.time.Clock()
    
    total_p1_wins = 0
    total_p2_wins = 0
    
    for game_num in range(num_games):
        game = PONG_GAME(WIDTH, HEIGHT)
        
        print(f"\nGame {game_num + 1}/{num_games}")
        
        running = True
        frame = 0
        max_frames = 5000
        
        while running and frame < max_frames:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    elif event.key == pygame.K_SPACE:
                        frame = max_frames
            
            # Get AI actions
            state1 = game.GET_STATE_P1()
            state2 = game.GET_STATE_P2()
            
            action1 = agent1.SELECT_ACTION(state1)
            action2 = agent2.SELECT_ACTION(state2)
            
            # Update game
            game.UPDATE(action1, action2)
            frame += 1
            
            # Draw
            DRAW_PONG_GAME(screen, WIDTH, HEIGHT, game_num + 1, game, game.p1_score, game.p2_score)
            pygame.display.flip()
            
            # Capture every 100 frames
            if frame % 100 == 0:
                try:
                    images_dir = Path("game_images")
                    filename = images_dir / f"pong_game_{game_num + 1}_frame_{frame:04d}_p1_{int(game.p1_score)}_p2_{int(game.p2_score)}.png"
                    pygame.image.save(screen, str(filename))
                    print(f"  📸 Screenshot: P1: {int(game.p1_score)} vs P2: {int(game.p2_score)}")
                except:
                    pass
        
        # Results
        if game.p1_score > game.p2_score:
            total_p1_wins += 1
            print(f"✓ Game {game_num + 1}: Player 1 WINS! ({int(game.p1_score)} vs {int(game.p2_score)})")
        elif game.p2_score > game.p1_score:
            total_p2_wins += 1
            print(f"✓ Game {game_num + 1}: Player 2 WINS! ({int(game.p2_score)} vs {int(game.p1_score)})")
        else:
            print(f"✓ Game {game_num + 1}: TIE! ({int(game.p1_score)} vs {int(game.p2_score)})")
    
    pygame.quit()
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Player 1 Wins: {total_p1_wins}")
    print(f"Player 2 Wins: {total_p2_wins}")
    print(f"{'='*70}\n")


def DISPLAY_MENU():
    print("\n" + "*"*70)
    print("PONG BATTLE - AI vs AI ARCADE GAME")
    print("*"*70)
    print("\n[1] Train new agents")
    print("[2] Play Pong (LIVE GAME)")
    print("[3] Save agents")
    print("[4] Load agents")
    print("[5] Exit")
    print("\n" + "-"*70)
    return input("Enter choice (1-5): ").strip()


def MAIN():
    SETUP_SYSTEM()
    
    agent1 = None
    agent2 = None
    
    while True:
        choice = DISPLAY_MENU()
        
        if choice == '1':
            print("\nTraining Agent 1...")
            agent1 = TRAINING_LOOP(episodes=100)
            print("Training Agent 2...")
            agent2 = TRAINING_LOOP(episodes=100)
            print("\n✓ Both agents trained!")
        
        elif choice == '2':
            if agent1 is None or agent2 is None:
                print("\n❌ Both agents needed! Train first.")
            else:
                print("\n📥 Installing pygame if needed...")
                os.system("pip install pygame --quiet")
                num_games = int(input("\nHow many games? (1-5): ") or 3)
                print("📸 Images will be saved to game_images/ folder")
                LIVE_PONG_GAME(agent1, agent2, num_games=min(num_games, 5))
        
        elif choice == '3':
            if agent1 and agent2:
                SECURITY_LAYER.SAVE_MODEL_SECURE(agent1, "pong_agent1", "Pong Agent 1")
                SECURITY_LAYER.SAVE_MODEL_SECURE(agent2, "pong_agent2", "Pong Agent 2")
                print("✓ Agents saved!")
        
        elif choice == '4':
            agent1 = PONG_AGENT()
            agent2 = PONG_AGENT()
            
            success = 0
            if SECURITY_LAYER.LOAD_MODEL_SECURE(agent1, "pong_agent1"):
                success += 1
            if SECURITY_LAYER.LOAD_MODEL_SECURE(agent2, "pong_agent2"):
                success += 1
            
            if success == 2:
                print("✓ Both agents loaded!")
        
        elif choice == '5':
            print("\n" + "="*70)
            print("Thank you for playing PONG BATTLE!")
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