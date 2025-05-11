import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from flappy_bird_env import FlappyBirdEnv
device = 'cuda' if torch.cuda.is_available() else 'cpu'

observation_size = 4
action_size = 2
learning_rate = 0.001
batch_size = 64

print(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1 = nn.Linear(observation_size, 128)
        self.lay2 = nn.Linear(128, 128)
        self.lay3 = nn.Linear(128, action_size)

    def forward(self,input):
        output = F.relu(self.lay1(input))
        output = F.relu(self.lay2(output))
        return self.lay3(output)

class DQNAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.gamma = 0.99 #discount factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.99  # chậm hơn để khám phá nhiều hơn ban đầu
        self.epsilon_min = 0.001   # vẫn cho phép một chút khám phá sau này
        self.target_update = 10  # Update target network every N episodes

        self.memory = ReplayMemory(20000)
        #policy network
        self.policy_net = DQNnetwork().to(device)
        #target y for trainning
        self.target_net = DQNnetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode

        # Define optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            torch.FloatTensor(state).to(device),
            torch.tensor([action]).to(device),
            torch.FloatTensor(next_state).to(device) if not done else None,
            torch.tensor([reward], dtype=torch.float32).to(device),
            torch.tensor([done], dtype=torch.bool).to(device)
        )


    def replay(self):
        if len(self.memory) < batch_size:
            return
    
        batch_data = self.memory.sample(batch_size)
        batch_data = Transition(*zip(*batch_data))
    
        # Tách dữ liệu
        batch_state = torch.stack(batch_data.state).to(device)                 # [batch_size, state_dim]
        batch_action = torch.cat(batch_data.action).unsqueeze(1).to(device)   # [batch_size, 1]
        batch_reward = torch.cat(batch_data.reward).to(device)                # [batch_size]
        
        # Xử lý next_state
        non_final_mask = ~torch.tensor(batch_data.done, dtype=torch.bool, device=device)
        non_final_next_states = torch.stack([s for s, d in zip(batch_data.next_state, batch_data.done) if not d]).to(device)
    
        # Tính giá trị Q hiện tại theo hành động đã chọn
        state_action_values = self.policy_net(batch_state).gather(1, batch_action)  # [batch_size, 1]
    
        # Tính Q mục tiêu
        next_state_values = torch.zeros(batch_size, device=device)
        if non_final_mask.any():
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    
        expected_state_action_values = batch_reward + self.gamma * next_state_values
    
        # Tính loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Tối ưu
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

    
    def save(self, filename):
        torch.save({
            'policy_model': self.policy_net.state_dict(),
            'target_model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    
    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_model'])
            self.target_net.load_state_dict(checkpoint['target_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filename}")
        else:
            print(f"No model found at {filename}")    


def train_dqn(episodes=1000, render_freq=100):
    """Train the DQN agent"""
    env = FlappyBirdEnv(render_mode="human" if render_freq > 0 else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent()
    scores = []
    epsilon_history = []
    
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Set render mode for visualization
        if e % render_freq == 0:
            env = FlappyBirdEnv(render_mode="human")
        else:
            env = FlappyBirdEnv(render_mode=None)
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train the network
            agent.replay()
        
        # Update target network
        if e % agent.target_update == 0:
            agent.update_target_network()
        
        # Save scores and epsilon
        scores.append(total_reward)
        epsilon_history.append(agent.epsilon)
        
        # Print progress
        print(f"Episode: {e}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically
        if e % 100 == 0:
            agent.save(f"flappy_bird_dqn_ep{e}.pth")
    
    # Save final model
    agent.save("flappy_bird_dqn_final.pth")
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history)
    plt.title('Epsilon per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('flappy_bird_dqn_training.png')
    plt.show()
    
    return agent




def test_agent(model_path="flappy_bird_dqn_final.pth", episodes=10):
    """Test the trained DQN agent"""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        return

    env = FlappyBirdEnv(render_mode="human")

    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.01  # Minimal exploration during testing

    for e in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Test Episode: {e}/{episodes}, Score: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    agent = train_dqn(episodes=1000, render_freq=100)
    # agent.load("flappy_bird_dqn_final.pth")
    test_agent(model_path="flappy_bird_dqn_final.pth", episodes=10)