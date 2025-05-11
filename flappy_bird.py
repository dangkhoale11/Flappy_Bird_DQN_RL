import pygame
import random
import sys

class FlappyBird:
    def __init__(self, x=600, y=400, bird=(30,30), pipe=(50, 90), tick=60, timer_spawn_pipe=1500):
        pygame.init()
        pygame.display.set_caption("Khoa")
        self.x = x
        self.y = y
        self.bird = bird
        self.pipe = pipe
        self.tick = tick
        self.timer_spawn_pipe = pygame.USEREVENT + 1
        pygame.time.set_timer(self.timer_spawn_pipe, timer_spawn_pipe)

        self.game_over = False
        self.fly = False
        self.velocity = 0

        self.clock = pygame.time.Clock()
        self.pipe_speed = 2.5
        self.pipes = []

        self.bird_gravity = 0.3
        self.bird_jump = -3

        self.screen = pygame.display.set_mode((x, y))
        self.flappy_bird_img = pygame.image.load("img/bird1.png").convert_alpha()
        self.background_img = pygame.image.load("img/bg.png").convert()
        self.floor_img = pygame.image.load("img/ground.png").convert()
        self.pipe_img = pygame.image.load("img/pipe.png").convert_alpha()
        self.pipe2_img = pygame.transform.flip(self.pipe_img, False, True)

        self.font = pygame.font.Font('font/Pixeltype.ttf', 50)

        self.background_img = pygame.transform.scale(self.background_img, (self.x, self.y))
        self.floor_img = pygame.transform.scale(self.floor_img, (self.x, 300))
        self.flappy_bird_img = pygame.transform.scale(self.flappy_bird_img, (self.bird[0], self.bird[1]))
        self.flappy_bird_rect = self.flappy_bird_img.get_rect(center=(self.x // 6, self.y // 2))

    def draw_background(self):
        self.screen.blit(self.background_img, (0, 0))
        self.screen.blit(self.floor_img, (0, 300))

    def draw_bird(self):
        self.screen.blit(self.flappy_bird_img, self.flappy_bird_rect)

    def pipe_spawn(self):
        if len(self.pipes) < 3:
            gap = self.pipe[1]
            pipe1_height = random.randint(50, 150)

            pipe1_img = pygame.transform.smoothscale(self.pipe_img, (self.pipe[0], pipe1_height))
            pipe1_rect = pipe1_img.get_rect(midbottom=(self.x, 300))

            pipe2_height = 300 - pipe1_height - gap
            pipe2_img = pygame.transform.smoothscale(self.pipe2_img, (self.pipe[0], pipe2_height))
            pipe2_rect = pipe2_img.get_rect(midtop=(self.x, 0))

            self.pipes.append((pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, False))  # False = chưa vượt qua

    def pipe_move(self):
        new_pipes = []
        for pipe in self.pipes:
            pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, passed = pipe
            pipe1_rect.centerx -= self.pipe_speed
            pipe2_rect.centerx -= self.pipe_speed

            if pipe1_rect.centerx > -50:
                new_pipes.append((pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, passed))
        self.pipes = new_pipes

    def draw_pipes(self):
        for pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, _ in self.pipes:
            self.screen.blit(pipe1_img, pipe1_rect)
            self.screen.blit(pipe2_img, pipe2_rect)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            if self.game_over:
                self.reset_game()
            elif not self.fly:
                self.fly = True
                self.velocity = self.bird_jump
            else:
                self.velocity = self.bird_jump

    def check_collision(self):
        for _, pipe1_rect, _, pipe2_rect, _ in self.pipes:
            if self.flappy_bird_rect.colliderect(pipe1_rect) or self.flappy_bird_rect.colliderect(pipe2_rect):
                self.game_over = True
                return True

        if self.flappy_bird_rect.bottom >= 300:
            self.flappy_bird_rect.bottom = 300
            self.game_over = True
            return True
        
        if self.flappy_bird_rect.top <= 0:
            self.game_over = True
            return True  # Chim đã va chạm với đỉnh

        if self.flappy_bird_rect.top <= 0:
            self.flappy_bird_rect.top = 0
            self.velocity = 0

        return False

    def check_passed_pipes(self):
        for i, pipe in enumerate(self.pipes):
            pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, passed = pipe
            if not passed and pipe1_rect.centerx < self.flappy_bird_rect.centerx:
                # Chim đã vượt qua ống này
                self.pipes[i] = (pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, True)
                print("Bird passed a pipe!")  # hoặc tăng điểm/reward tại đây

    def draw_game_over(self):
        text = self.font.render("Game Over - Press SPACE", True, (255, 0, 0))
        rect = text.get_rect(center=(self.x // 2, self.y // 2))
        self.screen.blit(text, rect)

    def reset_game(self):
        self.game_over = False
        self.fly = False
        self.velocity = 0
        self.pipes = []
        self.flappy_bird_rect.center = (self.x // 6, self.y // 2)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == self.timer_spawn_pipe and not self.game_over and self.fly:
                self.pipe_spawn()

        self.handle_input()

        if not self.game_over and self.fly:
            self.velocity += self.bird_gravity
            self.flappy_bird_rect.y += self.velocity
            self.check_collision()
            self.pipe_move()
            self.check_passed_pipes()

        self.draw_background()
        self.draw_bird()
        self.draw_pipes()

        if self.game_over:
            self.draw_game_over()

        pygame.display.update()
        self.clock.tick(self.tick)

if __name__ == "__main__":
    game = FlappyBird()
    while True:
        game.update()
