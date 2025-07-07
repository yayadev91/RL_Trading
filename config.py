# config.py

# ========= ENVIRONNEMENT RL =========
# (utilisé par: rl_trading/envs/trading_env.py)
INITIAL_QUANTITY = 0
INITIAL_CASH = 100
REWARD_TYPE = "delta_pnl"       # delta_pnl, sharpe, price_change, etc.
ACTION_SPACE_TYPE = "discrete3" # discrete3, continuous, ...
commission_rate = 0.001
MAX_DRAWDOWN = None
SELL_FRAC = 1.0
BUY_FRAC = 1.0 #fraction of the total cash available you use
FEATURE_IDX = None


# ========= DATA SOURCES =========
# Pour YFinance

SOURCE = "yahoo"

YAHOO_SYMBOL = "SPY"     # Format Yahoo
YAHOO_TIMEFRAME = "1d"       # Format Yahoo (1h, 1d, etc.)
YAHOO_HISTORY = "1000d"       # Format Yahoo (100d, 2y...)
YAHOO_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", 
    "1h", "1d", "5d", "1wk", "1mo", "3mo"
]
YAHOO_PERIODS = [
    "1d", "5d", "7d", "1mo", "3mo", 
    "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
]


# Pour Binance

BINANCE_SYMBOL = "BTCUSDT"
BINANCE_TIMEFRAME = "4h"
BINANCE_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
     "6h", "8h", "12h", "1d", "3d", "1w", "1M"
]
BINANCE_HISTORY_DAYS = 100
FEATURES = ['Open','High','Low','Close','Volume',
    'sma_20',
    'sma_50',
    'rsi_14',
    'macd',
    'macd_signal',
    'boll_up',
    'boll_down',
    'atr_14',
    'stoch_k',
    'obv',
    'returns_5', 'returns_10',
    'momentum_10',
    'vol_ratio',
    'vol_delta_5', 'vol_delta_20',
    'williams_r',
    'cmf_20',
    'price_above_sma20',
    'daily_range',
    'close_open_ratio',
    'zscore_50',
    'amp_range_20',
    'skew_20',
    'kurt_20',
]


    
DEVICE = "cuda"

# Période de backtest hors échantillon
BACKTEST_START   = "2019-02-01"  # date de début (YYYY-MM-DD)
BACKTEST_END     = "2021-07-10"  # date de fin   (YYYY-MM-DD)
MAX_WINDOW = 50# fenetre de rolling max de certaines features

# ========= MODELE RL =========
# Q-Network (modèle) - à adapter selon ton archi

penalty_coeff = 0.96 # penalité sur la reward pour hold and buy si lactif descend
bonus_coeff = 0.08 #bonus pour la vente si l'actif descend.
Q_INPUT_DIM = len(FEATURES)       # ex: nombre de features de l'obs, à adapter dynamiquement si besoin
#Q_HIDDEN_DIMS = [128, 64]  # liste des tailles de couches cachées (MLP de base)
Q_OUTPUT_DIM = 3         # nombre d’actions (ex: Buy, Hold, Sell)
Q_DROPOUT = 0.175
Q_EMBEDDING_DIM = 128    # si besoin (par exemple pour Transformer ou GNN)
Q_NUM_LAYERS = 1  
SEQUENCE_LENGTH = 6  # profondeur (ex: LSTM, transformer)
GAMMA = 1
EPSILON = 0.97
EPSILON_DECAY = 0.967
EPSILON_MIN = 0.05     
NUM_EPISODES = 120
EPISODE_LENGTH = 4
NHEAD = 2
LR = 5e-4
PATIENCE = 15
THRESHOLD = 1
# Ajoute ici tous les paramètres que tu veux rendre modifiables pour ton modèle


Q_MODEL_TYPES = ["MLP", "TRANSFORMER"]  # Pour plusieurs runs simultanés

# ===== DQN Replay Buffer =====

BUFFER_SIZE = 100_000  # Capacité maximale du buffer (nombre de transitions)
BATCH_SIZE = 160    # Nombre de transitions tirées au hasard à chaque entraînement

# ========= AGENT RL / TRAINING =========
# (utilisé par: rl_trading/agents/)
RL_ALGOS = ["PPO", "DQN"]  # ou ["PPO", "DQN", "A2C", ...]
N_RUNS_PER_ALGO = 3        # Pour faire la moyenne sur 
                        #plusieurs runs (stabilité)
#EPOCHS = 700




