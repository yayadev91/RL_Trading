import numpy as np
import torch
from agents.dqn_agent import DQNAgent, get_q_model
from envs.trading_env import TradingEnv
from data.feature_pipeline import FeaturePipeline
from models.transformer_q import TransformerQNet
from config import *
import pickle


def train_val_test_split(data, train_size=0.7, val_size=0.15):
    n = len(data)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    train = data[:n_train]
    val = data[n_train:n_train+n_val]
    test = data[n_train+n_val:]
    return train, val, test



def run_episode(env, agent, training=True):
    """
    Exécute un épisode complet sur l'env donné.
    Si training=True, on collecte transitions et on apprend, sinon éval greedy.
    """
    prev_eps = agent.epsilon
    #print(prev_eps)
    if training:
        agent.q_net.train()
    else:
        agent.epsilon = 0.0
        agent.q_net.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state)
        #print(action)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if training:
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
        state = next_state

    if training:
        agent.update_target()
    agent.epsilon = prev_eps
    agent.q_net.train()
    return total_reward

def train():
    # Préparation des données
    pipeline = FeaturePipeline()
    data = pipeline.fit_transform(pipeline.get_yahoo_data()).to_numpy().astype(float)
    
    train_data, val_data, test_data = train_val_test_split(data, 0.7, 0.15)

    env_train = TradingEnv(train_data, reward_type=REWARD_TYPE)
    env_val   = TradingEnv(val_data,   reward_type=REWARD_TYPE)
    env_test  = TradingEnv(test_data,  reward_type=REWARD_TYPE)

    # 3) créer agent

    model = get_q_model("TRANSFORMER")
    agent = DQNAgent(model)

    # 4) boucle d’entraînement simplifiée
    all_train_rewards = []
    all_val_rewards = []
    best_val = 0
    patience = PATIENCE
    counter = 0
    best_model_path = "best_model_parameters.pt"
    for ep in range(NUM_EPISODES):
        print(env_train.data[0])
        state, _ = env_train.reset(); done=False
        train_rewards = []
        
        while not done:
            action = agent.select_action(state)
            ns, r, done, *_ = env_train.step(action)
            agent.store_transition(state, action, r, ns, done)
            agent.train_step()
            state = ns
            train_rewards.append(r)
        

        agent.update_target()
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        val_reward = run_episode(env_val, agent)  # on minimise -reward
        if (best_val == 0) or (val_reward > best_val) :
            best_val = val_reward
            torch.save(agent.q_net.state_dict(), best_model_path)
            print(f"==> Model saved at epoch {ep+1} with val_loss: {val_reward:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping au bout de {ep+1} épisodes !")
                break
        print(f"ep {ep+1}/{NUM_EPISODES} : train_reward : {sum(train_rewards)} val_reward : {val_reward}")
        all_train_rewards.append(sum(train_rewards))
        all_val_rewards.append(val_reward)
    # ——— Évaluation finale sur test ———
    test_ret = run_episode(env_test, agent, training=False)
    print(f"=== TEST FINAL ===\nTest reward: {test_ret:.4f}")

    # ——— Sauvegarde ———
    torch.save(agent.q_net.state_dict(), "best_model_parameters.pt")
    np.save("train_rewards.npy", np.array(all_train_rewards))
    np.save("val_rewards.npy",   np.array(all_val_rewards))
    np.save("test_reward.npy",   np.array([test_ret]))

if __name__ == "__main__":
    train()
    import matplotlib.pyplot as plt
    train_rewards = np.load("train_rewards.npy")
    val_rewards = np.load("val_rewards.npy")
    #test_rewards = np.load("test_reward.npy")
    plt.plot(train_rewards, label="Train")
    plt.plot(val_rewards, label="Val")
    #plt.plot(test_rewards, label="test")
    plt.legend()
    plt.show()
