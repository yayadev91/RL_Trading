# backtest.py

import numpy as np
import torch
import matplotlib
# utiliser Agg pour éviter les problèmes de backend non interactif
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from performance import compute_returns, compute_alpha_beta

from config import (
    YAHOO_SYMBOL, YAHOO_TIMEFRAME,
    BACKTEST_START, BACKTEST_END,
    REWARD_TYPE
)
from data.feature_pipeline import FeaturePipeline
from envs.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent, get_q_model

# 1) Charger les données de backtest hors-échantillon
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
def load_trained_agent(model_path: str = "best_model_parameters.pt") -> DQNAgent:
    model = get_q_model("TRANSFORMER")
    agent = DQNAgent(model)
    state_dict = torch.load(model_path)
    agent.q_net.load_state_dict(state_dict)
    agent.q_net.eval()
    agent.epsilon = 0.0  # exploitation pure dès le backtest
    return agent

# 4) Simuler le backtest et renvoyer la courbe d’équité
def run_backtest(agent: DQNAgent, data: np.ndarray):
    env = TradingEnv(data, reward_type=REWARD_TYPE)
    state, _ = env.reset()
    done = False
    equity = [env.portfolio_value]
    actions = []
    quantity = []
    cashs = []
    while not done:
        quantity.append(env.position)
        cashs.append(env.cash)
        action = agent.select_action(state)
        actions.append(action)
        state, reward, done, truncated, info = env.step(action)
        equity.append(info["portfolio_value"])
        
    return np.array(equity), np.array(actions), np.array(quantity), np.array(cashs)

# 5) Calculer les métriques de performance
def compute_metrics(equity: np.ndarray) -> dict:
    returns = pd.Series(equity).pct_change().dropna()
    cumulative = equity[-1] / equity[0] - 1
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(356)
    return {
        "cumulative_return": cumulative,
        "sharpe_ratio": sharpe
    }

# 6) Fonction main
def main():
    # Chargement & préparation 
    raw = load_backtest_data()
    data = prepare_data(raw)

    # Agent & backtest
    agent = load_trained_agent()
    equity_curve, actions, quantity, cashs = run_backtest(agent, data)

    # Alignement des dates
    dates = pd.to_datetime(raw["Date"].iloc[-len(equity_curve):]).reset_index(drop=True)
    prices = raw["Close"].iloc[-len(equity_curve):].reset_index(drop=True)


    #Test
    R_market   = compute_returns(prices["SPY"])
    #print(f"0{R_market}")
    R_strategy = compute_returns(equity_curve)
    #print(f"1{R_strategy}")
    alpha, beta, eps = compute_alpha_beta(R_strategy, R_market)
    print(f"Alpha: {alpha:.2%}, Beta: {beta:.2f}, Epsilon: {eps.mean():.2f}")
    
    # Metrics
        # Metrics
    metrics = compute_metrics(equity_curve)
    print(f"\n=== Performance Metrics ===")
    print(f"Cumulative Return   : {metrics['cumulative_return']*100:.2f}%")
    print(f"Annualized Sharpe   : {metrics['sharpe_ratio']:.2f}")

    # Alignement des dates
    dates = pd.to_datetime(raw["Date"].iloc[-len(equity_curve):]).reset_index(drop=True)

    # Courbe d’équité + Drawdown + Histogramme
    fig, axs = plt.subplots(4, 1, figsize=(12, 14),
                        gridspec_kw={"height_ratios": [2, 1, 1, 1]})

    ## 1. Courbe d’équité
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(dates, equity_curve, marker="o", linestyle="-", color="#1a73e8", label="Equity Curve",alpha=0.7)
    ax2.plot(dates, prices,     linestyle="--", color="#f9a602", label="Price (Close)", alpha=0.7)
    ax1.set_ylabel("Portfolio Value")
    ax2.set_ylabel("Price")
    ax1.set_title("Courbe d’équité (bleu) vs Prix du sous-jacent (orange)", fontsize=15, fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)
    # Indices d'actions (attention : equity a une longueur de +1 par rapport à actions)
    print(actions)
    print(cashs)
    #buy_idx = [i  if actions[i] == 1 and cashs[i] != 0 else -1 for i in range(len(actions))]
    buy_idx = np.where((actions == 1) & (cashs != 0))[0]
    #buy_idx = [np.where(actions == 1 )[0] +1 if cashs != 0 else -1] # +1 car equity_curve est décalé (on ajoute la valeur après chaque step)
    sell_idx = np.where((actions == 2) & (quantity != 0))[0]
    print(f"Buying dates{buy_idx}")
    print(f"Selling dates{sell_idx}")
    print(f"quantité d'actifs{quantity}")
    print(f"quantité de cash{cashs}")
    print(f"equity_value{equity_curve}")
    # Ajoute les marqueurs à la courbe d’équité
    ax1.scatter(dates.iloc[buy_idx], equity_curve[buy_idx], color="green", label="Achat", marker="^", s=70, zorder=5)
    ax1.scatter(dates.iloc[sell_idx], equity_curve[sell_idx], color="red", label="Vente", marker="v", s=70, zorder=5)

    # Pour éviter le double label dans la légende
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")


    ## 2. Drawdown
    running_max = pd.Series(equity_curve).cummax()
    drawdown   = (equity_curve - running_max) / running_max
    axs[1].fill_between(dates, drawdown, color="#e84545", alpha=0.5, label="Drawdown")
    axs[1].set_title("Drawdown (%)", fontsize=14)
    axs[1].set_ylabel("Drawdown")
    axs[1].legend()
    axs[1].grid(True)

    ## 3. Comparaison des rendements journaliers
    axs[2].plot(dates[1:], R_market.iloc[1:].values,   label="Market Return",   alpha=0.7, linewidth=1.5)
    axs[2].plot(dates[1:], R_strategy.iloc[1:].values, label="Strategy Return", alpha=0.7, linewidth=1.5)
    axs[2].set_title("Comparison of Daily Returns", fontsize=14)
    axs[2].set_ylabel("Daily Return")
    axs[2].legend()
    axs[2].grid(True)

    ## 4. Histogramme des rendements de la stratégie
    returns = pd.Series(equity_curve).pct_change().dropna()
    axs[3].hist(returns, bins=40, color="#43aa8b", alpha=0.8, edgecolor="k")
    axs[3].set_title("Distribution des rendements journaliers", fontsize=14)
    axs[3].set_xlabel("Daily Return")
    axs[3].set_ylabel("Fréquence")
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("equity_backtest.png")
    plt.close(fig)
    print(f"\nGraphiques sauvegardés dans 'equity_backtest.png'")


if __name__ == "__main__":
    main()
