import gymnasium as gym
import numpy as np
import pygame
import time
from gymnasium.envs.registration import register

# Register the Flappy Bird environment
register(
    id="FlappyBird-v0",
    entry_point='flappy_bird_env:FlappyBirdEnv',
)


# Import after registration to ensure it works correctly
import flappy_bird_env as flappy_bird_env

def test_random_agent(episodes=5):
    """
    Test the environment with a random agent that selects actions randomly
    """
    # Create the environment
    env = gym.make("FlappyBird-v0", render_mode="human")
    
    for episode in range(episodes):
        # Reset the environment and get initial state
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{episodes}")
        print(f"Initial observation: {obs}")
        
        # Run until the episode is done
        while not done:
            # Choose a random action (0 = do nothing, 1 = flap)
            action = env.action_space.sample()
            
            # Take the action and get the new state
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update metrics
            total_reward += reward
            steps += 1
            
            # Print information every few steps
            if steps % 20 == 0:
                print(f"Step: {steps}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
                print(f"Observation: {obs}")
            
            # Ensure the environment renders
            env.render()
            
            # Small delay to make it visually easier to follow
            time.sleep(0.01)
        
        print(f"Episode {episode+1} finished after {steps} steps with total reward: {total_reward:.2f}")
    
    env.close()

def test_controlled_agent(episodes=3):
    """
    Test with a simple controlled agent that flaps when the bird is above a certain height
    """
    env = gym.make("FlappyBird-v0", render_mode="human")
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{episodes} (Controlled Agent)")
        
        while not done:
            # Simple height-based policy:
            # If bird is above the center of the gap, don't flap (action 0)
            # If bird is below the center of the gap, flap (action 1)
            bird_y = obs[0]  # Bird's y position
            gap_center = (obs[3] + obs[4]) / 2  # Center of pipe gap
            
            # Simple decision: flap if below center of gap
            action = 1 if bird_y > gap_center else 0
            
            # Take action and get new state
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update metrics
            total_reward += reward
            steps += 1
            
            # Print information
            if steps % 20 == 0:
                print(f"Step: {steps}, Bird Y: {bird_y:.1f}, Gap Center: {gap_center:.1f}")
                print(f"Action: {'Flap' if action == 1 else 'Don\'t Flap'}, Reward: {reward:.2f}")
            
            env.render()
            time.sleep(0.01)
        
        print(f"Episode {episode+1} finished after {steps} steps with total reward: {total_reward:.2f}")
    
    env.close()

def manual_control():
    """
    Allow the user to control the bird manually with the SPACE key
    """
    env = gym.make("FlappyBird-v0", render_mode="human")
    obs, info = env.reset()
    
    print("Manual Control Mode")
    print("Press SPACE to flap, Q to quit")
    
    running = True
    total_reward = 0
    steps = 0
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Get current key state
        keys = pygame.key.get_pressed()
        action = 1 if keys[pygame.K_SPACE] else 0
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update metrics
        total_reward += reward
        steps += 1
        
        # Display information
        if steps % 30 == 0:
            print(f"Step: {steps}, Total Reward: {total_reward:.2f}")
        
        # Reset if episode ended
        if done:
            print(f"Game over! Score: {info.get('score', 0)}, Steps: {steps}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            steps = 0
        
        # Render
        env.render()
        time.sleep(0.01)
    
    env.close()

if __name__ == "__main__":
    print("\nFlappy Bird Environment Test")
    print("1: Random Agent")
    print("2: Simple Controlled Agent")
    print("3: Manual Control")
    
    choice = input("Select test mode (1-3): ")
    
    if choice == "1":
        test_random_agent()
    elif choice == "2":
        test_controlled_agent()
    elif choice == "3":
        manual_control()
    else:
        print("Invalid choice. Running random agent test by default.")
        test_random_agent()