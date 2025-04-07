import numpy as np
import random

# Define the environment (4x4 grid)
num_states = 16
num_actions = 4
q_table = np.zeros((num_states, num_actions))

# Define the parameters
alpha = .1
gamma = .9
epsilon = .2
num_episodes = 1000

# Define a simple reward structure
rewards = np.zeros(num_states)
rewards[15] = 1

def get_next_state(state, action):
    if action == 0 and state >=4:
        return state - 4
    elif action == 1 and (state + 1) % 4 != 0:
        return state + 1
    elif action == 2 and state < 12:
        return state + 4
    elif action == 3 and state % 4 != 0:
        return state - 1
    else: 
        return state

# Q-Learning algorithm
for episode in range(num_episodes):
    state = random.randint(0, num_states - 1)
    while state != 15:
      if random.uniform(0, 1) < epsilon:  
        action = random.randint(0, num_actions - 1)
      else:
          action = np.argmax(q_table[state])
      
      next_state = get_next_state(state, action)       
      reward = rewards[next_state]
      old_value = q_table[state, action]
      next_max = np.max(q_table[next_state])
      
      # Q-Learning update rule
      new_value = old_value + alpha + (reward + gamma * next_max - old_value)
      q_table[state, action] =  new_value
      
      state = next_state

# Display the learned Q-table
print("Learned Q-Table:")
print(q_table)            
             