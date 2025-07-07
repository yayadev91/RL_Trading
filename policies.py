# policies.py
import torch

def always_hold_policy(obs,**kwargs):
    return 0  # 0 = Hold

def always_buy_policy(obs,**kwargs):
    return 1  # 1 = Buy

def always_sell_policy(obs,**kwargs):
    return 2  # 2 = Sell

def random_policy(obs, env=None,**kwargs):
    return env.action_space.sample()

class RLPolicy:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def __call__(self, obs, **kwargs):
        # obs : numpy array (env format)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action