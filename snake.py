from __future__ import annotations

import pygame
import numpy as np


# Define colors
BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
RED = np.array((255, 0, 0))
GREEN = np.array((0, 255, 0))
BLUE = np.array((0, 0, 255))

# Initialize Pygame
pygame.init()


class FPSDisplay:
    def __init__(self, game:Game):
        self.game = game
        self.font = pygame.font.SysFont(None, 25)

    def draw(self):
        fps_text = self.font.render(f"FPS: {int(self.game.clock.get_fps())}({self.game.fps})", True, WHITE)
        self.game.window.blit(fps_text, (10, 10))



class Game:

    CELL_SIZE = 30
    _GRID_SIZE = 20

    GRID_SIZE = np.array((_GRID_SIZE, _GRID_SIZE))
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE

    DIRECTIONS = ['LEFT', 'UP', 'RIGHT', 'DOWN']
    DIRECTIONS_VALS = [np.array(i) for i in ([-1, 0], [0, -1], [1, 0], [0, 1])]

    APPLE_COLOR = BLUE
    SNAKE_COLOR = WHITE

    def __init__(self):
        self.window = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Snake Game")

        self.clock = pygame.time.Clock()

        self.fps = 10
        self.fps_display = FPSDisplay(self)

        self.running = True

        self.init_new_game()

        self.to_call_on_draw = [self.fps_display.draw]
        self.to_call_on_game_quit = []

    def init_new_game(self):
        self.snake_size = 1
        self.snake_positions = [self.get_random_pos()]
        self.direction = 0

        self.new_apple()
        
    def get_random_pos(self):
        return np.random.randint(0, self._GRID_SIZE, size=(2,))
        
    def new_apple(self):
        self.apple_pos = self.get_random_pos()

        while self.intersects_body(self.apple_pos):
            self.apple_pos = self.get_random_pos()

    def turn_right(self):
        self.direction = (self.direction+1) % 4

    def turn_left(self):
        self.direction = (self.direction-1) % 4

    @property
    def current_move_direction(self):
        return self.DIRECTIONS_VALS[self.direction]
    
    def intersects_body(self, position):
        return any(np.array_equal(position, pos) for pos in self.snake_positions)

    def move(self):
        next_pos = (self.snake_positions[0] + self.current_move_direction)%self.GRID_SIZE

        if self.intersects_body(next_pos):
            self.init_new_game()
            return False

        self.snake_positions.insert(0, next_pos)

        if (self.apple_pos == self.snake_positions[0]).all():
            self.snake_size += 1
            self.new_apple()

        if len(self.snake_positions) > self.snake_size:
            self.snake_positions.pop()

        return True

    def draw(self):
        CELL_SIZE = self.CELL_SIZE
        pygame.draw.rect(self.window, self.APPLE_COLOR, (*self.apple_pos*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for position in self.snake_positions:
            pygame.draw.rect(self.window, self.SNAKE_COLOR, (*position*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        for func in self.to_call_on_draw:
            func()

    def step_game_loop(self, allow_events):
        global FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif allow_events and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.turn_left()
                elif event.key == pygame.K_RIGHT:
                    self.turn_right()

            elif event.type == pygame.MOUSEWHEEL:
                self.fps += event.y

        self.window.fill(BLACK)
        self.draw()
        pygame.display.update()
        self.clock.tick(self.fps)

    def run_loop(self):
        while self.running:
            game.move()
            game.step_game_loop(allow_events=True)

    def quit(self):
        self.running = False
        pygame.quit()

        for func in self.to_call_on_game_quit:
            func()


if __name__ == '__main__':
    game = Game()
    game.run_loop()
