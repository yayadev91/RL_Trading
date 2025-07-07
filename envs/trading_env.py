import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import *

class TradingEnv(gym.Env):
    def __init__(self, data, window_size=SEQUENCE_LENGTH, initial_cash=INITIAL_CASH,
     reward_type=REWARD_TYPE):
        super().__init__()
        self.qty =
        self.data = data  # numpy array ou DataFrame
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.reward_type = reward_type
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        # observation_space shape = (window_size, nb_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, data.shape[1]),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0    # 0 = pas d’action, 1 = en position
        self.entry_price = 0
        self.portfolio_value = self.cash
        self.done = False
        self.rewards = []
        return self._get_obs(), {}
    
    def _get_obs(self):
        return self.data[(self.current_step - self.window_size):self.current_step]


    def compute_reward(self, price, pv):
        # 1. Reward basée sur le changement de portefeuille (donc price)
        if self.reward_type == "delta_pnl":
            return pv - self.portfolio_value

        # 2. Reward basée sur le Sharpe ratio rolling
        elif self.reward_type == "sharpe":
            # On ajoute quand même le reward courant AVANT de calculer le Sharpe
            curr_reward = pv - self.portfolio_value
            tmp_rewards = self.rewards + [curr_reward]
            N = min(30, len(tmp_rewards))
            returns = np.array(tmp_rewards[-N:]) if N > 0 else np.array([0])
            mean = returns.mean()
            std = returns.std() + 1e-8
            sharpe = mean / std
            return sharpe

        # 3. Reward custom selon le prix direct
        elif self.reward_type == "price_change":
            # Ex: simple delta de prix entre step t et t-1
            return price - self.last_price


    

    def step(self, action, buy_frac=1.0, sell_frac=1.0):
        """
        action = 1 : achète buy_frac de la trésorerie disponible
        action = 2 : vend sell_frac de la position détenue
        """
        price = self.data[self.current_step, 0]
        print(f"prix{price}")

        # Achat
        if action == 1 and self.cash > 0:
            amount_to_spend = self.cash * buy_frac
            shares_to_buy = amount_to_spend / price
            self.position += shares_to_buy
            print(f"erreur2{amount_to_spend}")
            self.cash -= amount_to_spend

        # Vente
        elif action == 2 and self.position > 0:
            shares_to_sell = self.position * sell_frac
            amount_received = shares_to_sell * price
            self.position -= shares_to_sell
            self.cash += amount_received

        # Avance d'un pas
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True

        # Nouvelle valeur du portefeuille (pour l'info)
        pv = self.cash + self.position * price
        info = {"portfolio_value": pv}

        # Optionnel : reward peut être la variation de portefeuille
        reward = pv - self.portfolio_value
        self.portfolio_value = pv

        return self._get_obs(), reward, self.done, False, info





