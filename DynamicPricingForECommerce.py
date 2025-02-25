import pandas as pd
import numpy as np
import random

class DynamicPricingForECommerceEnv:
    def __init__(self, df, product_num, price_num):
        self.df = df.copy()
        self.n_products = product_num
        self.n_prices = price_num
        self.state = np.zeros((self.n_products, 4))
        self.price_ranges = self.df["price_range"].tolist()
        
    def reset(self):
        self.state = np.zeros((self.n_products, 4))
        for i in range(self.n_products):
            self.state[i, 0] = 0  # Initial price index
            #self.state[i, 1] = np.random.choice([0, 1])  # Random weekday (0) or weekend (1)
            #self.state[i, 1] = self.df.loc[i, "Week"] # Random weekday (0) or weekend (1)
            self.state[i, 2] = self.df.loc[i, "Impression"]  # Base demand as previous demand
            self.state[i, 3] = self.df.loc[i, "elasticity"]  # Store elasticity
        return self.state.copy()

    #action: the price to execute
    def step(self, actions):
        rewards = []
        new_state = self.state.copy()
        
        for i, action in enumerate(actions):
            price = self.price_ranges[i][action]
            D0 = self.state[i, 2]  # Previous demand
            P0 = self.state[i, 0]  # Previous price index
            #E = self.state[i, 3]  # Elasticity
            C = self.df.loc[i, "cost"]  # Cost remains fixed
            slope = self.df.loc[i, "slope"]
            mean_quantity = self.df.loc[i, "quantity_mean"]
            E = (slope)*(price/mean_quantity)
            
            demand = D0 + (D0 * E / P0) * (price - P0) if P0 > 0 else D0  # Avoid division by zero
            demand = max(demand, 0)
            reward = (price - C) * demand
            rewards.append(reward)
            
            new_state[i, 0] = action  # Update price index
            new_state[i, 2] = demand  # Update demand based on new price
            #new_state[i, 1] = 1 - self.state[i, 1]  # Toggle weekday/weekend
        
        total_reward = sum(rewards)
        self.state = new_state.copy()
        return new_state, total_reward, False

    