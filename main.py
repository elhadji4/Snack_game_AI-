import sys
import time
import random
import pygame
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtCore import Qt
from collections import deque

# 🚀 Initialisation de Pygame
pygame.init()

# 📌 Paramètres du jeu
WIDTH, HEIGHT = 500, 500
GRID_SIZE = 20
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
OBSTACLE_COLOR = (128, 128, 128)  # Gray color for obstacles

# 🐍 Classe Snake
class Snake:
    def __init__(self):
        self.body = [[100, 100]]  # Position initiale du serpent
        self.direction = "RIGHT"  # Direction initiale
        self.speed = 10  # Initial speed (frames per second)

    def move(self):
        """Déplace le serpent dans la direction actuelle"""
        head = self.body[0].copy()
        if self.direction == "UP":
            head[1] -= GRID_SIZE
        elif self.direction == "DOWN":
            head[1] += GRID_SIZE
        elif self.direction == "LEFT":
            head[0] -= GRID_SIZE
        elif self.direction == "RIGHT":
            head[0] += GRID_SIZE
        self.body.insert(0, head)

    def grow(self):
        """Fait grandir le serpent (appelé lorsqu'il mange de la nourriture)"""
        self.body.append(self.body[-1])  # Ajoute un nouveau bloc au corps du serpent
        self.speed += 0.5  # Increase speed as the snake grows

    def check_collision(self):
        """Vérifie si le serpent entre en collision avec lui-même ou les murs"""
        head = self.body[0]
        if (head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT):
            return True
        if head in self.body[1:]:
            return True
        return False

    def change_direction(self, new_direction):
        """Change la direction du serpent"""
        if new_direction == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif new_direction == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif new_direction == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif new_direction == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

# 🍎 Classe SnakeFood
class SnakeFood:
    def __init__(self):
        self.position = self.generate_food()

    def generate_food(self):
        """Génère une nouvelle position pour la nourriture"""
        return [random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE,
                random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE]

    def respawn(self):
        """Change la position de la nourriture"""
        self.position = self.generate_food()

# 🚧 Classe Obstacle
class Obstacle:
    def __init__(self):
        self.position = self.generate_obstacle()

    def generate_obstacle(self):
        """Génère une nouvelle position pour l'obstacle"""
        return [random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE,
                random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE]

    def respawn(self):
        """Change la position de l'obstacle"""
        self.position = self.generate_obstacle()

# 🎮 Classe GameBoard
class GameBoard:
    def __init__(self):
        self.snake = Snake()
        self.food = SnakeFood()
        self.obstacles = [Obstacle() for _ in range(3)]  # Add 3 obstacles
        self.score = 0
        self.done = False

    def reset(self):
        """Réinitialise le jeu"""
        self.snake = Snake()
        self.food.respawn()
        for obstacle in self.obstacles:
            obstacle.respawn()
        self.score = 0
        self.done = False
        return self._get_observation()

    def step(self, action):
        """Met à jour l'état du jeu en fonction de l'action"""
        if action == 0:  # UP
            self.snake.change_direction("UP")
        elif action == 1:  # DOWN
            self.snake.change_direction("DOWN")
        elif action == 2:  # LEFT
            self.snake.change_direction("LEFT")
        elif action == 3:  # RIGHT
            self.snake.change_direction("RIGHT")
        self.snake.move()

        # Calculer la distance entre la tête du serpent et la nourriture
        head = self.snake.body[0]
        food = self.food.position
        distance_to_food = np.sqrt((head[0] - food[0]) ** 2 + (head[1] - food[1]) ** 2)

        # Vérifier si le serpent mange la nourriture
        if head == self.food.position:
            self.score += 1
            self.food.respawn()
            self.snake.grow()
            reward = 20  # Récompense pour avoir mangé la nourriture
        else:
            self.snake.body.pop()
            # Récompense basée sur la distance à la nourriture
            reward = -0.1 * distance_to_food  # Pénalité pour s'éloigner de la nourriture

        # Vérifier les collisions avec les obstacles
        for obstacle in self.obstacles:
            if head == obstacle.position:
                self.done = True
                reward = -100  # Pénalité pour collision avec un obstacle

        # Vérifier les collisions avec le corps ou les murs
        if self.snake.check_collision():
            self.done = True
            reward = -100  # Pénalité pour collision

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        """Retourne l'état actuel du jeu sous forme d'observation"""
        head = self.snake.body[0]
        food = self.food.position
        direction = self.snake.direction

        # Distance aux murs
        distance_to_walls = [
            head[0] / WIDTH,
            (WIDTH - head[0]) / WIDTH,
            head[1] / HEIGHT,
            (HEIGHT - head[1]) / HEIGHT,
        ]

        # Distance à la nourriture
        direction_to_food = [
            (food[0] - head[0]) / WIDTH,
            (food[1] - head[1]) / HEIGHT,
        ]

        # Distance aux obstacles
        distance_to_obstacles = []
        for obstacle in self.obstacles:
            distance_to_obstacles.append((obstacle.position[0] - head[0]) / WIDTH)
            distance_to_obstacles.append((obstacle.position[1] - head[1]) / HEIGHT)

        # Observation complète
        observation = np.array(
            [head[0] / WIDTH, head[1] / HEIGHT] +
            [food[0] / WIDTH, food[1] / HEIGHT] +
            distance_to_walls + direction_to_food +
            distance_to_obstacles +
            [direction == "UP", direction == "DOWN", direction == "LEFT", direction == "RIGHT"],
            dtype=np.float32
        )
        return observation

    def render(self):
        """Affiche le jeu à l'écran"""
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        screen.fill(BLACK)

        for block in self.snake.body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(block[0], block[1], GRID_SIZE, GRID_SIZE))

        pygame.draw.rect(screen, RED, pygame.Rect(self.food.position[0], self.food.position[1], GRID_SIZE, GRID_SIZE))

        for obstacle in self.obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, pygame.Rect(obstacle.position[0], obstacle.position[1], GRID_SIZE, GRID_SIZE))

        pygame.display.flip()

    def close(self):
        pygame.quit()

# 🎮 Classe SnakeEnv (Custom Gym Environment)
class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)  # Updated shape to (20,)
        self.game = GameBoard()

    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.step(action)

    def render(self, mode="human"):
        self.game.render()

    def close(self):
        self.game.close()

# 🏆 🧠 Mode IA: Entraînement du Snake avec Reinforcement Learning
def train_snake():
    """Fonction pour entraîner l'IA avec DQN"""
    env = SnakeEnv()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        exploration_fraction=0.2,
        exploration_final_eps=0.1,
        buffer_size=100000,
        batch_size=64,
    )

    for match in range(5):
        print(f"🏋️ Entraînement de l'IA (Match {match + 1}/5)...")
        model.learn(total_timesteps=20000)
        model.save(f"snake_ai_match_{match + 1}")

    print("🎉 Entraînement terminé !")
    return model

# 📌 PyQt Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake Game")
        self.setGeometry(100, 100, 400, 300)

        # Layout
        layout = QVBoxLayout()

        # Bouton Commencer
        self.start_button = QPushButton("Commencer")
        self.start_button.clicked.connect(self.start_game)
        layout.addWidget(self.start_button)

        # Bouton Réinitialiser
        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.clicked.connect(self.reset_game)
        layout.addWidget(self.reset_button)

        # Bouton Quitter
        self.quit_button = QPushButton("Quitter")
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button)

        # Score Label
        self.score_label = QLabel("Score: -")
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)

        # Conteneur principal
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Variables pour les scores
        self.human_scores = []
        self.ai_scores = []

    def start_game(self):
        """Démarre la série de matchs et compare les scores"""
        QMessageBox.information(self, "Entraînement", "L'IA est en train de s'entraîner...")

        # Entraînement de l'IA
        model = train_snake()

        # Réinitialiser les scores
        self.human_scores = []
        self.ai_scores = []

        # Jouer 10 matchs
        for match in range(10):
            first_player = random.choice(["Humain", "IA"])
            QMessageBox.information(self, "Démarrage", f"{first_player} joue en premier !")

            if first_player == "IA":
                ai_score = self.play_ai(model)
                self.ai_scores.append(ai_score)
                human_score = self.play_human()
                self.human_scores.append(human_score)
            else:
                human_score = self.play_human()
                self.human_scores.append(human_score)
                ai_score = self.play_ai(model)
                self.ai_scores.append(ai_score)

            result = f"Match {match + 1}:\nScore Humain: {human_score}\nScore IA: {ai_score}\n"
            winner = "Égalité !" if human_score == ai_score else ("Le joueur humain a gagné !" if human_score > ai_score else "L'IA a gagné !")
            QMessageBox.information(self, "Résultat du Match", result + winner)

            if match < 9:
                reply = QMessageBox.question(self, "Continuer", "Voulez-vous continuer ?", QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    break

        self.display_final_results()

    def play_human(self):
        """Le joueur humain joue une partie et retourne son score"""
        pygame.init()
        game = GameBoard()
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and game.snake.direction != "DOWN":
                        game.snake.change_direction("UP")
                    elif event.key == pygame.K_DOWN and game.snake.direction != "UP":
                        game.snake.change_direction("DOWN")
                    elif event.key == pygame.K_LEFT and game.snake.direction != "RIGHT":
                        game.snake.change_direction("LEFT")
                    elif event.key == pygame.K_RIGHT and game.snake.direction != "LEFT":
                        game.snake.change_direction("RIGHT")

            observation, reward, done, info = game.step(None)
            game.render()

            if done:
                print(f"Game Over! Score: {game.score}")
                running = False

            clock.tick(game.snake.speed)  # Use snake's speed for game clock

        game.close()
        return game.score

    def play_ai(self, model):
        """L'IA joue une partie et retourne son score"""
        env = SnakeEnv()
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.1)
        ai_score = env.game.score
        env.close()
        return ai_score

    def display_final_results(self):
        """Affiche les scores finaux et met à jour l'affichage"""
        total_human_score = sum(self.human_scores)
        total_ai_score = sum(self.ai_scores)
        final_result = f"Score Total Humain: {total_human_score}\nScore Total IA: {total_ai_score}\n"
        winner = "Égalité !" if total_human_score == total_ai_score else ("Le joueur humain remporte la série !" if total_human_score > total_ai_score else "L'IA remporte la série !")
        
        self.score_label.setText(f"Score Final - Humain: {total_human_score} | IA: {total_ai_score}")

        QMessageBox.information(self, "Résultats Finaux", final_result + winner)

    def reset_game(self):
        """Réinitialise le jeu et les scores"""
        self.human_scores = []
        self.ai_scores = []
        self.score_label.setText("Score: -")
        QMessageBox.information(self, "Réinitialisation", "Le jeu a été réinitialisé.")

# Lancer l'application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())