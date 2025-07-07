import numpy as np
from rl_trading.envs.trading_env import TradingEnv

# Dummy data : 1000 jours, 14 features
data = np.random.randn(1000, 14).astype(np.float32)
env = TradingEnv(data, window_size=30, reward_type="sharpe")  # ou "delta_pnl", etc

obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, _, info = env.step(action)
    total_reward += reward

print("Test terminé. Total reward :", total_reward)
print("Dernier portfolio value :", info["portfolio_value"])
