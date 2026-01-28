import random
import sys
import pygame

CELL_SIZE = 24
GRID_W = 25
GRID_H = 20

WINDOW_W = CELL_SIZE * GRID_W
WINDOW_H = CELL_SIZE * GRID_H

BG_COLOR = (14, 20, 24)
GRID_COLOR = (22, 32, 38)
SNAKE_COLOR = (78, 211, 134)
HEAD_COLOR = (52, 180, 110)
FOOD_COLOR = (235, 88, 88)
TEXT_COLOR = (230, 234, 238)

MOVE_EVENT = pygame.USEREVENT + 1


def random_food(snake_cells):
    while True:
        pos = (random.randrange(GRID_W), random.randrange(GRID_H))
        if pos not in snake_cells:
            return pos


def draw_grid(surface):
    for x in range(0, WINDOW_W, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_H))
    for y in range(0, WINDOW_H, CELL_SIZE):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_W, y))


def draw_cell(surface, cell, color):
    rect = pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)


def reset_game():
    start = (GRID_W // 2, GRID_H // 2)
    snake = [start, (start[0] - 1, start[1]), (start[0] - 2, start[1])]
    direction = (1, 0)
    queued_dir = direction
    food = random_food(snake)
    score = 0
    alive = True
    return snake, direction, queued_dir, food, score, alive


def main():
    pygame.init()
    pygame.display.set_caption("Snek")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 22)

    speed_ms = 120
    pygame.time.set_timer(MOVE_EVENT, speed_ms)

    snake, direction, queued_dir, food, score, alive = reset_game()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_r:
                    snake, direction, queued_dir, food, score, alive = reset_game()
                if event.key == pygame.K_UP:
                    queued_dir = (0, -1)
                elif event.key == pygame.K_DOWN:
                    queued_dir = (0, 1)
                elif event.key == pygame.K_LEFT:
                    queued_dir = (-1, 0)
                elif event.key == pygame.K_RIGHT:
                    queued_dir = (1, 0)

            if event.type == MOVE_EVENT and alive:
                # Prevent reversing into itself.
                if (queued_dir[0] != -direction[0]) or (queued_dir[1] != -direction[1]):
                    direction = queued_dir

                head_x, head_y = snake[0]
                new_head = (head_x + direction[0], head_y + direction[1])

                if (
                    new_head[0] < 0
                    or new_head[0] >= GRID_W
                    or new_head[1] < 0
                    or new_head[1] >= GRID_H
                    or new_head in snake
                ):
                    alive = False
                else:
                    snake.insert(0, new_head)
                    if new_head == food:
                        score += 1
                        food = random_food(snake)
                        if score % 5 == 0 and speed_ms > 60:
                            speed_ms -= 5
                            pygame.time.set_timer(MOVE_EVENT, speed_ms)
                    else:
                        snake.pop()

        screen.fill(BG_COLOR)
        draw_grid(screen)

        for idx, cell in enumerate(snake):
            color = HEAD_COLOR if idx == 0 else SNAKE_COLOR
            draw_cell(screen, cell, color)

        draw_cell(screen, food, FOOD_COLOR)

        score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_text, (8, 6))

        if not alive:
            msg = font.render("Game Over - Press R to restart", True, TEXT_COLOR)
            msg_rect = msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))
            screen.blit(msg, msg_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
