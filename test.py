from data.feature_pipeline import FeaturePipeline
from models.MLP import MLPQNet
from models.transformer_q import TransformerQNet
from agents.dqn_agent import DQNAgent
from envs.trading_env import TradingEnv
from config import *

# Préparation des datasets
datasets = {
    "yahoo": FeaturePipeline('yahoo', FEATURES),
    "binance": FeaturePipeline('binance', FEATURES)
}

data_dict = {}
for name, pipeline in datasets.items():
    if name == "yahoo":
        df = pipeline.get_yahoo_data(YAHOO_SYMBOL, YAHOO_HISTORY, YAHOO_TIMEFRAME)
    else:
        df = pipeline.get_binance_data(BINANCE_SYMBOL, BINANCE_TIMEFRAME, BINANCE_HISTORY_DAYS)
    data_dict[name] = pipeline.fit_transform(df).to_numpy().astype(float)

# Entraînement sur chaque dataset
results = {}
for name, data in data_dict.items():
    print(f"\n--- Entraînement sur {name} ---")
    env = TradingEnv(data, window_size=SEQUENCE_LENGTH)
    model = TransformerQNet()
    agent = DQNAgent(model=model)
    rewards = []
    losses = []
    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0 # Initialize episode loss
        step_count = 0 # To average loss over steps

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None: # Only add loss if a training step occurred
                episode_loss += loss
                step_count += 1
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        losses.append(avg_loss)
        print(f"Episode {ep+1}/{NUM_EPISODES} - Total Reward: {total_reward:.2f} - Average Loss: {avg_loss:.4f}")

    results[name] = rewards
    print(f"{name} - Moyenne des rewards par épisode : {sum(rewards)/len(rewards):.2f}")
    print(f"{name} - Moyenne des losses par épisode : {sum(losses)/len(losses):.2f}")

# Comparaison simple (print)
print("\nRésumé des performances :")
for name in results:
    print(f"{name} : Reward moyen sur {NUM_EPISODES} épisodes = {sum(results[name])/len(results[name]):.2f}")
