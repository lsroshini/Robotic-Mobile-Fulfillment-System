import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from typing import List, Dict, Tuple, Any

# Custom Warehouse Environment
class WarehouseEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(5)  # Example action space size
        self.observation_space = Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Example observation space
        self.distances: np.ndarray = np.random.rand(5, 5)  # Replace with your actual distances
        self.pheromone: np.ndarray = np.ones((5, 5))  # Initial pheromone levels
        self.state: np.ndarray = self.reset()[0]  # Call reset and get the initial state

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility
        self.state = np.zeros(5)  # Reset state
        return self.state, {}  # Return state and empty info dictionary

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Implement the logic for taking a step in the environment
        self.state = np.random.rand(5)  # Example state update; replace with your logic
        reward = np.random.rand()  # Replace with actual reward calculation
        terminated = False  # Set to True if the episode is done
        truncated = False  # Set to True if the episode was truncated due to time limits
        info = {}
        return self.state, reward, terminated, truncated, info

# Ant Colony Optimization (ACO) Implementation
class ACO:
    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.n_best = 3  # Number of best paths to consider
        self.best_paths = []  # Store best paths for visualization

    def run(self) -> List[int]:
        all_paths = [[0, 1, 2]]  # Example paths; replace with your actual path generation logic
        self.spread_pheromone(all_paths, self.n_best)
        self.best_paths = all_paths[:self.n_best]  # Store the best paths for visualization
        return all_paths[0]  # Return the best path for demonstration

    def spread_pheromone(self, all_paths: List[List[int]], n_best: int) -> None:
        for path in all_paths[:n_best]:
            for move in path:
                if self.env.distances[move].any():  # Check if distances[move] is non-zero
                    self.env.pheromone[move] += 1.0 / self.env.distances[move].sum()  # Spread pheromone

# Main Streamlit App
def main() -> None:
    st.title("Robotic Mobile Fulfillment System")
    st.sidebar.header("Control Panel")

    # Create environment and ACO instances
    env = WarehouseEnv()
    wrapped_env = Monitor(env)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # Initialize the model
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # ACO Run Button
    if st.sidebar.button("Run ACO"):
        aco_instance = ACO(env)  # Create ACO instance
        best_route = aco_instance.run()  # Run ACO to find best route
        st.write("Best route found:", best_route)
        
        # Visualize best paths if available
        if aco_instance.best_paths:
            visualize_paths(aco_instance.best_paths)
        else:
            st.write("No paths found for visualization.")

    # Learning Loop
    if st.sidebar.button("Train Model"):
        training_rewards = []  # List to store rewards during training
        timesteps = 10000  # Number of training timesteps

        # Custom callback to capture rewards during training
        class RewardCallback:
            def __init__(self):
                self.rewards = []

            def __call__(self, locals_, globals_):
                self.rewards.append(locals_['self'].rollout_buffer.rewards[-1])  # Capture last reward

        reward_callback = RewardCallback()

        model.learn(total_timesteps=timesteps, callback=reward_callback)
        st.success("Model trained successfully!")

        # Display training results
        if reward_callback.rewards:
            st.subheader("Training Rewards")
            st.line_chart(reward_callback.rewards)

def visualize_paths(paths: List[List[int]]) -> None:
    st.subheader("Best Paths Visualization")
    
    if not paths:  # Check if there are paths to visualize
        st.write("No paths to visualize.")
        return

    # Create a DataFrame for better handling
    import pandas as pd

    # Ensure paths are consistent and padded for heatmap
    max_length = max(len(path) for path in paths)
    padded_paths = [path + [None] * (max_length - len(path)) for path in paths]

    # Convert to DataFrame for seaborn compatibility
    path_df = pd.DataFrame(padded_paths)
    
    # Create a heatmap of pheromone levels
    plt.figure(figsize=(10, 6))
    sns.heatmap(path_df, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Pheromone Level'})
    plt.title("Best Paths")
    plt.xlabel("Path Index")
    plt.ylabel("Moves")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
