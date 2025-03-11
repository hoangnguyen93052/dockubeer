import random
import numpy as np
import math
import json
import pygame
from pygame.locals import *

# Game Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# AI Behavior States
IDLE = "idle"
PATROLLING = "patrolling"
CHASING = "chasing"
ATTACKING = "attacking"

class GameObject:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class NPC(GameObject):
    def __init__(self, x, y, width, height, color):
        super().__init__(x, y, width, height, color)
        self.state = IDLE
        self.patrol_path = [(x, y), (x + 100, y)]
        self.current_target = 0

    def update(self, player_pos):
        if self.state == IDLE:
            self.patrol()
        elif self.state == PATROLLING:
            self.patrol()
        elif self.state == CHASING:
            self.chase(player_pos)
        elif self.state == ATTACKING:
            self.attack()

    def patrol(self):
        target = self.patrol_path[self.current_target]
        if self.rect.x < target[0]:
            self.rect.x += 2
        elif self.rect.x > target[0]:
            self.rect.x -= 2
        else:
            self.current_target = (self.current_target + 1) % len(self.patrol_path)

        if self.rect.x != self.patrol_path[self.current_target][0]:
            self.state = PATROLLING
        else:
            self.state = IDLE

    def chase(self, player_pos):
        if self.rect.x < player_pos[0]:
            self.rect.x += 4
        else:
            self.rect.x -= 4

        if self.rect.colliderect(pygame.Rect(player_pos[0], player_pos[1], 10, 10)):
            self.state = ATTACKING

    def attack(self):
        print("NPC attacks!")

class Player(GameObject):
    def __init__(self, x, y, width, height, color):
        super().__init__(x, y, width, height, color)
    
    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

class Game:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("AI for Gaming")
        self.clock = pygame.time.Clock()
        self.npc = NPC(300, 300, 50, 50, GREEN)
        self.player = Player(100, 100, 10, 10, RED)
        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            self.player.move(-5, 0)
        if keys[K_RIGHT]:
            self.player.move(5, 0)
        if keys[K_UP]:
            self.player.move(0, -5)
        if keys[K_DOWN]:
            self.player.move(0, 5)

    def update(self):
        self.npc.update((self.player.rect.x + 5, self.player.rect.y + 5))

    def render(self):
        self.window.fill(WHITE)
        self.npc.draw(self.window)
        self.player.draw(self.window)
        pygame.display.flip()

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()