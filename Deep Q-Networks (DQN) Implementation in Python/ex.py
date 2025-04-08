import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

# Hyperparameters
env_name = 'CartPole-v1'
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000
batch_size = 32
epsilon = 0.1
target_update_frequency = 100

# Initialize environment and DQN
env = gym.make(env_name)
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Experience replay buffer
replay_buffer = []

def train(num_episodes):
    step_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(state, dtype=torch.float32, device=device)).argmax().item()

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state)
            total_reward += reward

            # Store experience
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)

            state = next_state

            # Sample from buffer and train
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                step_count += 1
                if step_count % target_update_frequency == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Train the agent
train(num_episodes=1000)
