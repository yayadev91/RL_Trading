# optuna_search.py
import optuna
from optuna.exceptions import TrialPruned

import torch
from config import *
from train import train_val_test_split, run_episode, FeaturePipeline, TradingEnv, DQNAgent, get_q_model

def objective(trial):
    # 1) proposer les hyper-params
    BATCH_SIZE = trial.suggest_int("batch_size", 64, 192, step=32)
    #Q_HIDDEN_DIMS = trial.suggest_categorical("q_hidden_dims", [[64, 32], [128, 64], [256, 128]])
    Q_NUM_LAYERS = trial.suggest_int("num_layers", 1,3)
    NHEAD         = trial.suggest_int("nhead", 2, 4)
    Q_DROPOUT     = trial.suggest_float("q_dropout", 0.01, 0.3)
    Q_EMBEDDING_DIM = trial.suggest_int("q_embedding_dim", 64, 128, step=32)
    SEQUENCE_LENGTH = trial.suggest_int("seq_length", 4, 20)
    LR = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    EPSILON = trial.suggest_float("epsilon",0.8, 1)
    EPSILON_DECAY = trial.suggest_float("epsilon_decay",0.90, 1)
    bonus_coeff = trial.suggest_float("bonus_vente", 0, 1)
    penalty_coeff = trial.suggest_float("penalise_achat", 0, 1)

    # (optionnel) réduire le nombre d'épisodes pour accélérer la recherche
    NUM_EPISODES  = 5

    # 2) préparer les données et envs
    pipeline = FeaturePipeline()
    data = pipeline.get_yahoo_data()
    data = pipeline.fit_transform(data).to_numpy().astype(float)
    train_data, val_data, _ = train_val_test_split(data)
    env_train = TradingEnv(train_data, reward_type=REWARD_TYPE)
    env_val   = TradingEnv(val_data,   reward_type=REWARD_TYPE)

    # 3) créer agent
    model = get_q_model("TRANSFORMER")
    agent = DQNAgent(model)

    # 4) boucle d’entraînement simplifiée
    for ep in range(NUM_EPISODES):
        state, _ = env_train.reset(); done=False
        while not done:
            action = agent.select_action(state)
            ns, r, done, *_ = env_train.step(action)
            agent.store_transition(state, action, r, ns, done)
            agent.train_step()
            state = ns

        agent.update_target()
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        
        val_loss = -run_episode(env_val, agent)  # on minimise -reward
        print(f"ep {ep} : on essaye de maximier ca reward_val : {-val_loss}")

        # rapport au pruner
        trial.report(val_loss, ep)
        if trial.should_prune():
            raise TrialPruned()

    return val_loss

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=5, timeout=3600)

    print("Best params:", study.best_trial.params)
     # Affiche la loss (val loss) associée
    print("Best reward :", -study.best_value)
