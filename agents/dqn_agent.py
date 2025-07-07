from collections import deque
import random
import numpy as np
from config import *
import torch
import numpy as np
from models.transformer_q import TransformerQNet  # Ou le modèle que tu veux
from agents.replay_buffer import ReplayBuffer



def get_q_model(model_type):
    if model_type.upper() == "MLP":
        return MLPQNet()
    elif model_type.upper() == "TRANSFORMER":
        return TransformerQNet()
    # elif model_type.upper() == "EMGNN":
    #     return EMGNNQNet()
    else:
        raise ValueError(f"Unknown model type {model_type}")


class DQNAgent:
    def __init__(self, model, device=DEVICE):
        self.device = device
        self.q_net = model.to(device)
        self.target_q_net = type(model)().to(device)  # Instancie un modèle de même type que q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer()
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(Q_OUTPUT_DIM)
        # Adapte ici si ton modèle attend du batch/seq (par exemple Transformer)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return q_vals.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch, 1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

   
        q_all = self.q_net(states)
        

        q_values = q_all.gather(1, actions)
       

        q_values = q_values.squeeze()
        

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
        

        target = rewards + self.gamma * next_q_values * (1 - dones)
        

        loss = torch.nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



    def val_step(self):
        """
        Tire un batch du replay buffer, calcule la loss Q-learning
        en mode évaluation (no grad / pas de backward), et renvoie
        la valeur scalaire de la loss pour monitoring.
        """
        # 1) Si pas assez de samples, on skip
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return None

        # 2) Récupère un batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # 3) Passage en tenseurs
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch,1)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # 4) Calcul des Q-values et de la target, sans grad
        with torch.no_grad():
            # Q(s,a) actuel
            q_values = self.q_net(states).gather(1, actions).squeeze()

            # Q*(s',a') maxi via le target network
            next_q_values = self.target_q_net(next_states).max(dim=1)[0]

            # cible : r + γ * max_a' Q*(s',a') * (1-done)
            target = rewards + self.gamma * next_q_values * (1 - dones)

            # MSE entre Q(s,a) et target
            loss = torch.nn.functional.mse_loss(q_values, target)

        # 5) On ne fait ni backward ni update ici
        return loss.item()




    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())