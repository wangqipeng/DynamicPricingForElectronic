# Dynamic Retail Pricing via Q-Learning

## Introduction
This project implements a **dynamic retail pricing strategy** using **Q-learning**, a reinforcement learning algorithm that optimizes pricing decisions over time. The goal is to maximize revenue by selecting the best pricing strategy for each product based on market conditions, demand elasticity, and weekday trends.

## Dataset Description
This is a list of over 15,000 electronic products with pricing information across 10 unique fields provided by Datafiniti's Product Database. The dataset also includes the brand, category, merchant, name, source, and more.

Note that this is a sample of a large dataset. The full dataset is available through Datafiniti.

### **Download the Dataset**
To run this project, you need to download the dataset from **[Kaggle](https://www.kaggle.com/datasets/datafiniti/electronic-products-prices/data)** and place it in the current directory.

---

## Solution Overview
The **Q-learning agent** interacts with an environment where it adjusts product prices and learns from feedback (rewards). The environment simulates **consumer demand** by considering **price elasticity**, **weekday/weekend effects**, and other market factors.

### **Key Components**
1. **Actions** - The available pricing options for each product.
2. **States** - The conditions that influence pricing decisions.
3. **Reward Function** - The metric used to evaluate pricing performance.
4. **Q-learning Algorithm** - The reinforcement learning method used to optimize pricing over time.

---

## **1. Actions (Price Adjustments)**
Each product has a **discrete set of price options** (e.g., 10% lower, same, or 10% higher than the base price). The Q-learning agent selects one of these price levels at each step.

### **Example Price Range for a Product:**
| Action Index | Price ($) |
|-------------|----------|
| 0           | 50       |
| 1           | 75       |
| 2           | 100      |
| 3           | 125      |
| 4           | 150      |

The agent learns which **price level** maximizes **profit** under different conditions.

---

## **2. State Representation**
Each product's **state** consists of:
1. **Price Index**: The currently selected price level.
2. **Weekday Indicator**: Weekday (0) vs. Weekend (1).
3. **Previous Demand**: The observed demand at the last price level.
4. **Price Elasticity**: The sensitivity of demand to price changes.

### **Example State for a Product:**
| Feature           | Value |
|------------------|------|
| Price Index      | 2    |
| Weekday/Weekend | 1    |
| Previous Demand  | 120  |
| Price Elasticity | -1.3 |

Each **state transition** updates these values as the environment evolves.

---

## **3. Reward Function**
The reward function measures **profitability** based on the chosen price and resulting demand:

\[
Reward = (Price - Cost) /  Demand
\]

Where:
- **Price**: The selected price for the product.
- **Cost**: The production cost of the product.
- **Demand**: The number of units sold at the selected price.

The Q-learning agent aims to **maximize cumulative reward** over time.

---

## **4. Q-learning Algorithm**
Q-learning is a **model-free reinforcement learning** algorithm that iteratively improves pricing decisions by updating a **Q-table**. The Q-table stores the expected rewards for each **(state, action) pair**.

### **Q-learning Formula**
\[
Q(s, a) = Q(s, a) + \alpha \times [r + \gamma \times \max_{a'} Q(s', a') - Q(s, a)]
\]
Where:
- \(Q(s, a)\) → Current Q-value for state \(s\) and action \(a\)
- \(\alpha\) → Learning rate (how much new information overrides old knowledge)
- \(r\) → Immediate reward received from the action
- \(\gamma\) → Discount factor (importance of future rewards)
- \(\max_{a'} Q(s', a')\) → Best Q-value from the next state \(s'\)

### **Algorithm Steps:**
1. **Initialize** the Q-table with zeros.
2. **Select an action** using the **ε-greedy policy**:
   - With probability **ε**, pick a random action (exploration).
   - Otherwise, pick the action with the highest Q-value (exploitation).
3. **Observe the reward** and **next state**.
4. **Update the Q-table** using the Q-learning formula.
5. **Repeat until convergence** (i.e., the pricing strategy stabilizes).

---

## **5. Implementation Workflow**
### **Step 1: Load Dataset**
The dataset includes:
- Product IDs
- Historical Prices
- Demand Data
- Cost Information
- Weekday vs. Weekend Indicators

### **Step 2: Initialize Q-learning Agent & Environment**
- Set up **state representation**.
- Define **pricing actions**.
- Initialize the **Q-table**.

### **Step 3: Train the Q-learning Agent**
- Simulate pricing decisions over multiple episodes.
- Update **Q-values** based on observed **profitability**.

### **Step 4: Evaluate the Learned Policy**
- Extract **optimal pricing strategies** from the trained Q-table.
- Compare **agent decisions vs. fixed-price strategies**.

---

## **6. How to Run the Project**
### **Steps to Execute**
1. **Download the dataset** from Kaggle: [Electronic Products Prices](https://www.kaggle.com/datasets/datafiniti/electronic-products-prices/data)
2. **Save the dataset** in the project directory.
3. **Run the command**:
   ```sh
   poetry run python main.py
   ```

---

## **Conclusion**
This project demonstrates how **Q-learning** can optimize **dynamic retail pricing** by balancing **profitability and demand sensitivity**. By leveraging **reinforcement learning**, retailers can implement **adaptive pricing strategies** that respond to market trends in real-time.

