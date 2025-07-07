# RL_Trading/backtest.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.feature_pipeline import FeaturePipeline
from envs.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent, get_q_model
from config import *


# === 1. Importe ton environnement et policies ===
from envs.trading_env import TradingEnv
from policies import always_hold_policy, random_policy, always_buy_policy, always_sell_policy, RLPolicy

# === 2. Importe le modèle utilisé pour RL ===
from models.transformer_q import TransformerQNet   # ou le modèle que tu utilises

# === 3. Charger les données (adapte ici si besoin) ===
import numpy as np
def load_backtest_data():
    print(f"Téléchargement Yahoo Finance {YAHOO_SYMBOL}, "
          f"backtest de {BACKTEST_START} à {BACKTEST_END}, intervalle={YAHOO_TIMEFRAME}")
    return FeaturePipeline.get_yahoo_backtest(
        symbol=YAHOO_SYMBOL,
        start=BACKTEST_START,
        end=BACKTEST_END,
        interval=YAHOO_TIMEFRAME
    )

# 2) Préparer les features sur toute la période
def prepare_data(df: pd.DataFrame) -> np.ndarray:
    df_feat = FeaturePipeline().fit_transform(df).reset_index(drop=True)
    return df_feat.to_numpy().astype(float)

# 3) Charger l’agent entraîné
def load_trained_agent(model_path: str = "GOOD_ONE.pt") -> DQNAgent:
    model = get_q_model("TRANSFORMER")
    agent = DQNAgent(model)
    state_dict = torch.load(model_path)
    agent.q_net.load_state_dict(state_dict)
    agent.q_net.eval()
    agent.epsilon = 0.0  # exploitation pure dès le backtest
    return agent
data = load_backtest_data()
data = prepare_data(data)  # ADAPTE ce chemin à tes données réelles !

env = TradingEnv(data)

# === 4. Charger ton modèle RL ===

agent = load_trained_agent("GOOD_ONE.pt")
rl_policy = RLPolicy(agent.q_net)

# === 5. Backtest runner ===
def backtest_policy(env, policy, max_steps=2000, render=False, **policy_kwargs):
    obs = env.reset()
    done = False
    rewards, actions, positions, cashes = [], [], [], []
    step = 0

    while not done and step < max_steps:
        action = policy(obs, env=env, **policy_kwargs)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        positions.append(getattr(env, 'position', None))
        cashes.append(getattr(env, 'cash', None))
        if render:
            env.render()
        step += 1

    return {
        "rewards": np.array(rewards),
        "actions": np.array(actions),
        "positions": np.array(positions),
        "cashes": np.array(cashes),
    }

# === 6. Lance les backtests ===
results_hold = backtest_policy(env, always_hold_policy)
results_random = backtest_policy(env, random_policy)
results_buy = backtest_policy(env, always_buy_policy)
results_sell = backtest_policy(env, always_sell_policy)
results_rl = backtest_policy(env, rl_policy)

# === 7. Visualisation ===
plt.figure(figsize=(12, 6))
plt.plot(results_rl['cashes'], label='RL Agent')
plt.plot(results_hold['cashes'], label='Always Hold')
plt.plot(results_buy['cashes'], label='Always Buy')
plt.plot(results_sell['cashes'], label='Always Sell')
plt.plot(results_random['cashes'], label='Random')
plt.xlabel('Step')
plt.ylabel('Cash')
plt.title('Backtest - Cash Curve')
plt.legend()
plt.grid(True)
plt.show()
