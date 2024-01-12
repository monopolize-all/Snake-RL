from __future__ import annotations

import snake
import models
import torch
import numpy as np
import matplotlib.pyplot as plt


class StateValueOverlay:

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.game = agent.game

        self.state_values = np.ndarray((*self.agent.game.GRID_SIZE, 4), dtype=np.float32)

    def calculate_state_values(self):
        agent = self.agent
        game = self.game
        
        snake_body_pos_list = []
        apple_pos = game.apple_pos
        
        for x in range(game._GRID_SIZE):
            for y in range(game._GRID_SIZE):
                for move_direction in range(4):
                    snake_head_pos = np.array((x, y))
                    state = agent.get_state(snake_head_pos, snake_body_pos_list, apple_pos, move_direction)
                    self.state_values[x, y, move_direction] = agent.value_nn(state)
        
    def draw(self):
        """Only calculates and draws state values when fps betwee 0 and 20"""
        if not 0 < self.game.fps < 20:
            return
        
        self.calculate_state_values()

        max_value = np.amax(self.state_values)
        min_value = np.amin(self.state_values)
        mid_value = (max_value+min_value)/2

        CELL_SIZE = self.game.CELL_SIZE
        CELL_SIZE_BY_2 = CELL_SIZE//2
        TRIANGLE_HEIGHT = CELL_SIZE//4

        triangles_by_move_direction = np.array((
            ((0, CELL_SIZE_BY_2), (TRIANGLE_HEIGHT, CELL_SIZE_BY_2+TRIANGLE_HEIGHT), (TRIANGLE_HEIGHT, CELL_SIZE_BY_2-TRIANGLE_HEIGHT)),
            ((CELL_SIZE_BY_2, 0), (CELL_SIZE_BY_2+TRIANGLE_HEIGHT, TRIANGLE_HEIGHT), (CELL_SIZE_BY_2-TRIANGLE_HEIGHT, TRIANGLE_HEIGHT)),
            ((CELL_SIZE, CELL_SIZE_BY_2), (CELL_SIZE-TRIANGLE_HEIGHT, CELL_SIZE_BY_2+TRIANGLE_HEIGHT), (CELL_SIZE-TRIANGLE_HEIGHT, CELL_SIZE_BY_2-TRIANGLE_HEIGHT)),
            ((CELL_SIZE_BY_2, CELL_SIZE), (CELL_SIZE_BY_2+TRIANGLE_HEIGHT, CELL_SIZE-TRIANGLE_HEIGHT), (CELL_SIZE_BY_2-TRIANGLE_HEIGHT, CELL_SIZE-TRIANGLE_HEIGHT)),
        ))

        TRIANGLES_DISTANCE_SCALING = 0.5

        triangles_by_move_direction = (triangles_by_move_direction-(CELL_SIZE_BY_2, CELL_SIZE_BY_2)) * TRIANGLES_DISTANCE_SCALING + (CELL_SIZE_BY_2, CELL_SIZE_BY_2)

        for x in range(self.agent.game._GRID_SIZE):
            for y in range(self.agent.game._GRID_SIZE):
                max_value = np.amax(self.state_values[x, y, :])
                min_value = np.amin(self.state_values[x, y, :])
                mid_value = (max_value+min_value)/2
                for move_direction in range(4):
                    value = self.state_values[x, y, move_direction]
                    if value > mid_value:
                        triangle_color = snake.GREEN * (value-mid_value)/(max_value-mid_value)
                    else:
                        triangle_color = snake.RED * (value-mid_value)/(max_value-mid_value)
                    triangle_points = triangles_by_move_direction[move_direction] + (x*CELL_SIZE, y*CELL_SIZE)
                    snake.pygame.draw.polygon(self.game.window, triangle_color.astype(np.uint8), triangle_points)


class Agent:

    MODEL_HIDDEN_SIZE = 64
    LR = 1e-4
    GAMMA = 0.9
    EPOCHS = 100

    APPLE_EAT_REWARD = 5
    APPLE_NO_EAT_REWARD = -0.1
    SNAKE_DEATH_REWARD = -10

    PLOT_SAVE_GAME_INTERVAL = 50

    SNAKE_START_SIZE = 1

    def __init__(self) -> None:
        self.model_hidden_size = self.MODEL_HIDDEN_SIZE
        self.lr = self.LR
        self.gamma = self.GAMMA

    def init_game(self):
        self.game_metrics = {'score': [], 'turns_played': []}
        self.game = snake.Game()
        self.game.snake_size = self.SNAKE_START_SIZE
        self.state_value_overlay = StateValueOverlay(self)
        self.game.to_call_on_draw.append(self.state_value_overlay.draw)
        self.game.to_call_on_game_quit.append(self.on_quit)

    def init_model(self):
        self.value_nn = models.ValueNN(6, self.model_hidden_size, 1, self.lr,
                                model_name='snake_valueNN', load_saved_model=True)
        
        self.current_game = 0
        
    def on_quit(self):
        self.value_nn.save()
        self.plot_and_save_game_score()
        quit()
        
    def plot_and_save_game_score(self):
        if self.game_metrics['score']:
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(self.game_metrics['score'], c='blue', label = 'score')
            ax[0].legend()
            ax[0].set_ylim(bottom=0)
            #ax[0].ylim(0, 2*max(self.game_scores))

            ax[1].plot(self.game_metrics['turns_played'], c='blue', label='turns_played')
            tmp = [x/y for x, y in zip(self.game_metrics['turns_played'], self.game_metrics['score'])]
            ax[1].plot(tmp, c='orange', label='avg_turns_played_per_score')
            ax[1].legend()
            ax[1].set_ylim(bottom=0)


            plt.savefig('game scores.png')

    def get_state(self, snake_head_pos, snake_body_pos_list, apple_pos, move_direction):
        game = self.game

        snakes_forward_direction = game.DIRECTIONS_VALS[move_direction]
        snakes_left_direction = game.DIRECTIONS_VALS[(move_direction-1)%4]
        snakes_right_direction = game.DIRECTIONS_VALS[(move_direction+1)%4]

        apple_pos_list = [apple_pos,
                          apple_pos-2*np.array((apple_pos[0], 0)),
                          apple_pos-2*np.array((0, apple_pos[1])),
                          apple_pos+2*np.array((self.game._GRID_SIZE-apple_pos[0], 0)),
                          apple_pos+2*np.array((0, self.game._GRID_SIZE-apple_pos[1])),
                          ]

        apple_relative_head = apple_pos_list - snake_head_pos

        food_front = min(np.dot(apple_relative_head, snakes_forward_direction))/game._GRID_SIZE
        food_left = min(np.dot(apple_relative_head, snakes_left_direction))/game._GRID_SIZE
        food_right = min(np.dot(apple_relative_head, snakes_right_direction))/game._GRID_SIZE

        if snake_body_pos_list:
            body_front = min(np.dot(snake_body_pos_list, snakes_forward_direction))/game._GRID_SIZE
            body_left = min(np.dot(snake_body_pos_list, snakes_left_direction))/game._GRID_SIZE
            body_right = min(np.dot(snake_body_pos_list, snakes_right_direction))/game._GRID_SIZE
        else:
            body_front = body_left = body_right = 1

        return torch.Tensor((food_front, food_left, food_right, body_front, body_left, body_right))

    def train(self, game_count):
        """One epoch is one snake life"""
        self.current_game = 0
        game_steps = 0
        while self.current_game < game_count:
            game_steps += 1
            if not self.train_step():
                self.game.snake_size = self.SNAKE_START_SIZE

                self.current_game += 1

                self.game_metrics['turns_played'].append(game_steps)
                game_steps = 0

                if self.current_game % self.PLOT_SAVE_GAME_INTERVAL == 0:
                    self.plot_and_save_game_score()

    def train_step(self):
        """Returns False when snake dies else True"""
        game = self.game

        game.step_game_loop(allow_events=False)

        snake_head_pos = game.snake_positions[0]
        snake_body_pos_list = list(game.snake_positions[1:])
        apple_pos = game.apple_pos
        move_direction = game.direction

        snakes_forward_direction = game.DIRECTIONS_VALS[move_direction]
        snakes_left_direction = game.DIRECTIONS_VALS[(move_direction-1)%4]
        snakes_right_direction = game.DIRECTIONS_VALS[(move_direction+1)%4]

        snake_head_pos_on_move_forward = (snake_head_pos+snakes_forward_direction) % game._GRID_SIZE
        snake_head_pos_on_move_left = (snake_head_pos+snakes_left_direction) % game._GRID_SIZE
        snake_head_pos_on_move_right = (snake_head_pos+snakes_right_direction) % game._GRID_SIZE

        current_state = self.get_state(snake_head_pos, snake_body_pos_list, apple_pos, move_direction)

        state_after_actions = [
            self.get_state(snake_head_pos_on_move_forward, snake_body_pos_list, apple_pos, move_direction),
            self.get_state(snake_head_pos+snakes_left_direction, snake_body_pos_list, apple_pos, (move_direction-1)%4),
            self.get_state(snake_head_pos+snakes_right_direction, snake_body_pos_list, apple_pos, (move_direction+1)%4)
        ]

        V_S = self.value_nn(current_state)

        # print('current_state:', current_state)
        # print('current_value:', V_S.item())

        V_S_a = [
            self.value_nn(state_after_actions[0]),
            self.value_nn(state_after_actions[1]),
            self.value_nn(state_after_actions[2])
        ]

        R_S_a_1 = [
            self.APPLE_EAT_REWARD if (apple_pos == snake_head_pos_on_move_forward).all() else self.APPLE_NO_EAT_REWARD,
            self.APPLE_EAT_REWARD if (apple_pos == snake_head_pos_on_move_left).all() else self.APPLE_NO_EAT_REWARD,
            self.APPLE_EAT_REWARD if (apple_pos == snake_head_pos_on_move_right).all() else self.APPLE_NO_EAT_REWARD
        ]
        R_S_a_2 = [
            self.SNAKE_DEATH_REWARD if game.intersects_body(snake_head_pos_on_move_forward) else 0,
            self.SNAKE_DEATH_REWARD if game.intersects_body(snake_head_pos_on_move_left) else 0,
            self.SNAKE_DEATH_REWARD if game.intersects_body(snake_head_pos_on_move_right) else 0
        ]

        V_S_a_with_R_S = [R_S_a_1[i] + R_S_a_2[i] + self.gamma * V_S_a[i] for i in range(3)]

        self.value_nn.train_on_datapoint(V_S, max(V_S_a_with_R_S))

        action_probabilities = torch.softmax(torch.Tensor(V_S_a_with_R_S), 0).numpy()

        chosen_action = np.random.choice((0, 1, 2), p=action_probabilities)

        if chosen_action == 1: game.turn_left()
        elif chosen_action == 2: game.turn_right()

        game_score = game.snake_size
        if not game.move():
            self.game_metrics['score'].append(game_score)
            return False
        
        return True

if __name__ == '__main__':
    agent = Agent()
    agent.init_game()
    agent.init_model()

    agent.train(500)
