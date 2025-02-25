import numpy as np
import random
import config

class QLearningAgent:
    def __init__(self, env):
        self.n_products = env.n_products
        self.n_actions = env.n_prices
        self.price_ranges = env.price_ranges
        self.Q = [np.zeros(len(env.price_ranges[i])) for i in range(self.n_products)]

    def choose_actions(self, state):
        actions = []
        for i in range(self.n_products):
            price_index = int(state[i, 0])
            if random.random() < config.EPSILON:
                actions.append(random.randint(0, len(self.price_ranges[i]) - 1))
            else:
                actions.append(np.argmax(self.Q[i][price_index]))
        return actions
    
    def update(self, state, actions, reward, next_state):
        for i in range(self.n_products):
            action = actions[i]
            price_index = int(state[i, 0])
            q_current = self.Q[i][action] if isinstance(self.Q[i], np.ndarray) else 0
            
            next_price_index = int(next_state[i, 0])
            q_next = self.Q[i][next_price_index] if isinstance(self.Q[i], np.ndarray) and 0 <= next_price_index < len(self.Q[i]) else 0
            
            self.Q[i][action] += config.ALPHA * (reward + config.GAMMA * q_next - q_current)

        if config.EPSILON > config.EPSILON_MIN:
            config.EPSILON *= config.EPSILON_DECAY



    def train_q_learning(self, env, episodes=50):
        rewards_history = []
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(10):
                actions = self.choose_actions(state)
                next_state, reward, done = env.step(actions)
                self.update(state, actions, reward, next_state)
                state = next_state.copy()
                total_reward += reward
            rewards_history.append(total_reward)
            print(f"Episode {e} - Total Reward: {total_reward:.2f}")
        return rewards_history
