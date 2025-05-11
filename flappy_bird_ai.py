import argparse
import os
from flappy_bird_dqn import train_dqn, test_agent

def main():
    parser = argparse.ArgumentParser(description='Train or test DQN agent for Flappy Bird')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training or testing')
    parser.add_argument('--render_freq', type=int, default=100,
                        help='Frequency of rendering during training (0 to disable)')
    parser.add_argument('--model_path', type=str, default='flappy_bird_dqn_final.pth',
                        help='Path to save/load model')

    args = parser.parse_args()
    
    # ✅ Dùng đường dẫn đúng như người dùng nhập, không gắn thêm 'models/'
    model_path = args.model_path

    if args.mode == 'train':
        print(f"Training for {args.episodes} episodes, rendering every {args.render_freq} episodes")
        train_dqn(episodes=args.episodes, render_freq=args.render_freq, model_path=model_path)
    else:
        print(f"Testing model from {model_path} for {args.episodes} episodes")
        test_agent(model_path=model_path, episodes=args.episodes)

if __name__ == "__main__":
    main()
