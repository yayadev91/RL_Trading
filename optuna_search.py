# optuna_search.py
import optuna
from optuna.exceptions import TrialPruned
from models.transformer_q import TransformerQNet

import torch
from config import REWARD_TYPE, Q_INPUT_DIM, Q_OUTPUT_DIM,PATIENCE,EPSILON_MIN, THRESHOLD, DEVICE
from train import train_val_test_split, run_episode, FeaturePipeline, TradingEnv, DQNAgent, get_q_model

def objective(trial):
    # 1) proposer les hyper-params
    # 1) proposer les hyper-params (tout en minuscules)
    batch_size       = trial.suggest_int("batch_size", 64, 192, step=32)
    q_num_layers     = trial.suggest_int("num_layers", 1, 5)
    nhead            = trial.suggest_int("nhead", 2, 4, step=2)
    q_dropout        = trial.suggest_float("q_dropout", 0.01, 0.3)
    q_embedding_dim  = trial.suggest_int("q_embedding_dim", 64, 128, step=32)
    sequence_length  = trial.suggest_int("seq_length", 4, 20)
    lr               = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    epsilon          = trial.suggest_float("epsilon", 0.8, 1)
    epsilon_decay    = trial.suggest_float("epsilon_decay", 0.90, 1)
    bonus_vente      = trial.suggest_float("bonus_vente", 0, 1)
    penalise_achat   = trial.suggest_float("penalise_achat", 0, 1)
    sell_frac        = trial.suggest_float("sell_frac", 0, 0.1)
    buy_frac         = trial.suggest_float("buy_frac", 0, 0.1)

    # 2) préparer les données et envs
    pipeline   = FeaturePipeline()
    data       = pipeline.get_yahoo_data()
    data       = pipeline.fit_transform(data).to_numpy().astype(float)
    train_data, val_data, _ = train_val_test_split(data)

    env_train = TradingEnv(
        train_data,
        reward_type=REWARD_TYPE,
        window_size=sequence_length  # utilisation de la variable lowercase
    )
    env_val   = TradingEnv(
        val_data,
        reward_type=REWARD_TYPE,
        window_size=sequence_length
    )

    # 3) créer agent avec le TransformerQNet
    model = TransformerQNet(
        input_dim=Q_INPUT_DIM,       # reste en majuscule, c’est issu du config
        seq_len=sequence_length,     # lowercase
        d_model=q_embedding_dim,     # lowercase
        nhead=nhead,                 # lowercase
        num_layers=q_num_layers,     # lowercase
        output_dim=Q_OUTPUT_DIM,     # issu du config
        dropout=q_dropout            # lowercase
    ).to(DEVICE)
    agent = DQNAgent(model)

    best_val = None
    patience = PATIENCE
    counter = 0

    # **Nom unique pour chaque trial**
    best_model_path = f"best_model_trial_{trial.number}.pt"
    NUM_EPISODES = 25
    # 4) boucle d’entraînement simplifiée
    for ep in range(NUM_EPISODES):
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
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * epsilon_decay)

        
        val_reward = run_episode(env_val, agent)  # on minimise -reward
        print(f"ep {ep+1} : on essaye de maximier ca reward_val : {val_reward}")
        if (best_val is None) or (val_reward > (best_val + THRESHOLD)):
            best_val = val_reward
            torch.save(agent.q_net.state_dict(), best_model_path)
            print(f"==> Model saved at epoch {ep+1} with val_reward: {val_reward:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping au bout de {ep+1} épisodes !")
                break
        print(f"ep {ep+1}/{NUM_EPISODES} : train_reward : {sum(train_rewards)} val_reward : {val_reward}")

        # rapport au pruner
        trial.report(val_reward, ep)
        if trial.should_prune():
            raise TrialPruned()

    return val_reward

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=15, timeout=2500)
    best_params = study.best_trial.params

    import json
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Best params:", study.best_trial.params)
     # Affiche la loss (val loss) associée
    print("Best reward :", study.best_value)
