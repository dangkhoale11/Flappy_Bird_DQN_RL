import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from flappy_bird import FlappyBird
from gymnasium.envs.registration import register

class FlappyBirdEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        self.render_mode = render_mode
        self.game = FlappyBird()
        self.action_space = gym.spaces.Discrete(2)  # 0: do nothing, 1: flap


        # Observation space: [bird_y,velocity, pipe_nearest_top_height, pipe_nearest_bottom_height]
        low = np.array([0, -10, 300 - self.game.pipe[1] - 150, 50], dtype=np.float32)
        high = np.array([300, 10,300 - self.game.pipe[1] - 50, 150], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.window = None
        self.score = 0
        self.steps_survived = 0
    def get_observation(self):
        bird_y = self.game.flappy_bird_rect.centery / 300
        velocity = self.game.velocity / 10
        
        nearest_pipe = None
        min_distance = float('inf')
        for pipe in self.game.pipes:
            pipe_x = pipe[1].centerx
            distance = pipe_x - self.game.flappy_bird_rect.centerx
            if distance > -50 and distance < min_distance:
                nearest_pipe = pipe
                min_distance = distance
        
        if nearest_pipe:
            pipe_top = nearest_pipe[3].height / 300
            pipe_bottom = (300 - nearest_pipe[1].height) / 300
        else:
            pipe_top = 0
            pipe_bottom = 1
        
        obs = np.array([bird_y, velocity, pipe_top, pipe_bottom], dtype=np.float32)
        return obs

    def step(self, action):
        reward = 0.1  # Thưởng nhỏ cho mỗi bước sống sót
        terminated = False
        truncated = False
        info = {}
        self.steps_survived += 1

        # Xử lý hành động
        if action == 1:
            self.game.velocity = self.game.bird_jump
        else:
            self.game.velocity += self.game.bird_gravity

        self.game.flappy_bird_rect.centery += self.game.velocity
            
        # Tạo ống
        if self.steps_survived % 150 == 0:
            self.game.pipe_spawn()
        
        if len(self.game.pipes) == 0:
            self.game.pipe_spawn()
            
        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return self.get_observation(), 0, True, False, {}

        self.game.pipe_move()

        # Thưởng khi vượt qua ống
        for i, pipe in enumerate(self.game.pipes):
            pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, passed = pipe
            if not passed and pipe1_rect.centerx < self.game.flappy_bird_rect.centerx:
                self.game.pipes[i] = (pipe1_img, pipe1_rect, pipe2_img, pipe2_rect, True)
                reward += 10  # Thưởng lớn hơn khi vượt qua ống
                self.score += 1

        # Kiểm tra va chạm
        collision = self.game.check_collision()
        if collision:
            terminated = True
            self.steps_survived = 0
            reward -= 20  # Phạt nặng hơn khi va chạm

        # Thưởng khi ở vị trí tốt
        for pipe in self.game.pipes:
            if not pipe[4]:  # Nếu chưa vượt qua ống
                pipe_top = pipe[3].height
                pipe_bottom = 300 - pipe[1].height
                bird_y = self.game.flappy_bird_rect.centery
                optimal_y = (pipe_top + pipe_bottom) / 2
                distance = abs(bird_y - optimal_y)
                reward += max(0, 1 - distance/100)  # Thưởng khi gần vị trí tối ưu

        if self.render_mode == "human":
            self.render()

        return self.get_observation(), reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset the game
        self.game.reset_game()
        self.game.fly = True  # Start the game immediately
        
        # Reset metrics
        self.score = 0
        
        
        # Generate initial pipes to make observation meaningful
        self.game.pipe_spawn()
        
        return self.get_observation(), {}

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Process pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Update game visuals
            self.game.draw_background()
            self.game.draw_bird()
            self.game.draw_pipes()
            
            # Display score
            score_text = self.game.font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.game.screen.blit(score_text, (10, 10))
            
            # Update display and control frame rate
            pygame.display.update()
            self.game.clock.tick(self.metadata["render_fps"])
        
        elif self.render_mode == "rgb_array":
            # Update game visuals
            self.game.draw_background()
            self.game.draw_bird()
            self.game.draw_pipes()
            
            # Return RGB array of game screen
            return pygame.surfarray.array3d(pygame.display.get_surface())

    def close(self):
        """Clean up resources"""
        pygame.quit()            

register(
    id="FlappyBird-v0",
    entry_point='flappy_bird_env:FlappyBirdEnv',
)
