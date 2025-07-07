from sklearn.preprocessing import MinMaxScaler
from config import *
import numpy as np
import pandas as pd

class FeaturePipeline:
    def __init__(self, dataset_type = SOURCE, features = FEATURES):
        self.dataset_type = dataset_type
        self.features = features
        self.scaler = MinMaxScaler()

    @staticmethod
    def get_yahoo_data(symbol=YAHOO_SYMBOL, period=YAHOO_HISTORY,
    interval=YAHOO_TIMEFRAME):
        print(f"Téléchargement Yahoo Finance {symbol}, période={period}, intervalle={interval}")
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        df = df.reset_index()
        return df
    @staticmethod
    def raw_yahoo_data(symbol=YAHOO_SYMBOL, period=YAHOO_HISTORY,
    interval=YAHOO_TIMEFRAME):
        print(f"Téléchargement Yahoo Finance {symbol}, période={period}, intervalle={interval}")
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        df = df.reset_index(drop =True)
        return df["Close"].to_numpy().astype(float)
        
    @staticmethod
    def get_yahoo_backtest(symbol=YAHOO_SYMBOL, start=BACKTEST_START,
    end=BACKTEST_END, interval=YAHOO_TIMEFRAME):
        """
        Télécharge les données pour backtest hors échantillon.
        Par défaut, utilise BACKTEST_START et BACKTEST_END depuis config.py.
        """
        print(f"Téléchargement Yahoo Finance {symbol}, backtest de {start} à {end}, intervalle={interval}")
        import yfinance as yf
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True
        )
        df = df.reset_index()
        return df

    @staticmethod
    def get_binance_data(symbol=BINANCE_SYMBOL, timeframe=BINANCE_TIMEFRAME, days=BINANCE_HISTORY_DAYS):
        print(f"Téléchargement Binance {symbol}, timeframe={timeframe}, sur {days} jours")
        from binance.client import Client
        import datetime
        client = Client()
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(days=days)
        klines = client.get_historical_klines(
            symbol, timeframe,
            start_time.strftime("%d %b %Y %H:%M:%S"),
            end_time.strftime("%d %b %Y %H:%M:%S")
        )
        # Parsing en DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        df = df.rename(columns={
            'open_time': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df.reset_index(drop=True)
        return df

    #Selection et creation des features
    

    def fit_transform(self, df):
        if self.dataset_type == 'yahoo':
            df = self._process_yahoo(df)
        elif self.dataset_type == 'binance':
            df = self._process_binance(df)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        df_selected = df[self.features].copy()
        # Normalisation MinMax sur toutes les colonnes/features
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_selected), columns=df_selected.columns, index=df_selected.index)
        return df_scaled

    def _process_yahoo(self, df):
    # Ajoute ici tes features custom Yahoo

    
        # 1. Moyenne mobile courte (SMA 20 déjà présente)
        if 'sma_20' in self.features:
            df['sma_20'] = df['Close'].rolling(window=20).mean().shift(1)
        # 2. Moyenne mobile longue
        if 'sma_50' in self.features:
            df['sma_50'] = df['Close'].rolling(window=50).mean().shift(1)
        # 3. RSI (Relative Strength Index)
        if 'rsi_14' in self.features:
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_down = down.rolling(14).mean()
            rs = roll_up / (roll_down + 1e-9)
            df['rsi_14'] = (100.0 - (100.0 / (1.0 + rs))).shift(1)
        # 4. MACD (ligne rapide)
        if 'macd' in self.features:
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = (ema12 - ema26).shift(1)
        # 5. MACD signal (ligne lente)
        if 'macd_signal' in self.features:
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            df['macd_signal'] = macd.ewm(span=9, adjust=False).mean().shift(1)
        # 6. Bollinger Bands (écart haut)
        if 'boll_up' in self.features:
            sma20 = df['Close'].rolling(20).mean()
            std20 = df['Close'].rolling(20).std()
            df['boll_up'] = (sma20 + 2 * std20).shift(1)
        # 7. Bollinger Bands (écart bas)
        if 'boll_down' in self.features:
            sma20 = df['Close'].rolling(20).mean()
            std20 = df['Close'].rolling(20).std()
            df['boll_down'] = (sma20 - 2 * std20).shift(1)
        # 8. ATR (Average True Range)
        if 'atr_14' in self.features:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean().shift(1)
        # 9. Stochastic Oscillator %K
        if 'stoch_k' in self.features:
            low14 = df['Low'].rolling(14).min()
            high14 = df['High'].rolling(14).max()
            df['stoch_k'] = (100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)).shift(1)
        # 10. OBV (On Balance Volume)
        if 'obv' in self.features:
            direction = np.where(df['Close'].diff() >= 0, 1, -1)
            obv = (df['Volume'] * direction).cumsum()
            df['obv'] = obv.shift(1)

        # 11. Returns sur plusieurs horizons (shift pour ne pas utiliser le close du jour t)
        if 'returns_5' in self.features:
            df['returns_5'] = df['Close'].pct_change(5).shift(1)
        if 'returns_10' in self.features:
            df['returns_10'] = df['Close'].pct_change(10).shift(1)

        # 12. Momentum (différence de prix sur x jours)
        if 'momentum_10' in self.features:
            df['momentum_10'] = (df['Close'] - df['Close'].shift(10)).shift(1)

        # 13. Volatility Ratio (rolling std court/long)
        if 'vol_ratio' in self.features:
            std_short = df['Close'].rolling(10).std()
            std_long = df['Close'].rolling(50).std()
            df['vol_ratio'] = (std_short / (std_long + 1e-9)).shift(1)

        # 14. Change de volume (shift pareil)
        if 'vol_delta_5' in self.features:
            df['vol_delta_5'] = df['Volume'].pct_change(5).shift(1)
        if 'vol_delta_20' in self.features:
            df['vol_delta_20'] = df['Volume'].pct_change(20).shift(1)

        # 15. Williams %R
        if 'williams_r' in self.features:
            high14 = df['High'].rolling(14).max()
            low14 = df['Low'].rolling(14).min()
            df['williams_r'] = (-100 * (high14 - df['Close']) / (high14 - low14 + 1e-9)).shift(1)

        # 16. Chaikin Money Flow
        if 'cmf_20' in self.features:
            mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9) * df['Volume']
            df['cmf_20'] = mfv.rolling(20).sum().shift(1) / df['Volume'].rolling(20).sum().shift(1)

        # 17. Price above/below SMA20 (gap par rapport à la moyenne)
        if 'price_above_sma20' in self.features:
            sma20 = df['Close'].rolling(20).mean().shift(1)
            df['price_above_sma20'] = (df['Close'] - sma20) / (sma20 + 1e-9)

        # 18. Daily Range (volatilité intra-day) — pas de shift, l'open/high/low sont connus à t
        if 'daily_range' in self.features:
            df['daily_range'] = (df['High'] - df['Low']) / df['Open']

        # 19. Close/Open ratio — pas de shift, open et close du jour dispo en fin de séance
        if 'close_open_ratio' in self.features:
            df['close_open_ratio'] = df['Close'] / (df['Open'] + 1e-9)

        # 20. Z-score du prix (vs. moyenne mobile 50)
        if 'zscore_50' in self.features:
            mean_50 = df['Close'].rolling(50).mean().shift(1)
            std_50 = df['Close'].rolling(50).std().shift(1)
            df['zscore_50'] = (df['Close'] - mean_50) / (std_50 + 1e-9)

        # 21. Amplitude range (high-low sur 20 jours)
        if 'amp_range_20' in self.features:
            df['amp_range_20'] = (df['High'].rolling(20).max() - df['Low'].rolling(20).min()).shift(1)

        # 22. Skewness rolling (sur 20 jours)
        if 'skew_20' in self.features:
            df['skew_20'] = df['Close'].rolling(20).apply(lambda x: ((x - np.mean(x))**3).mean() / (np.std(x)+1e-9)**3).shift(1)

        # 23. Kurtosis rolling (sur 20 jours)
        if 'kurt_20' in self.features:
            df['kurt_20'] = df['Close'].rolling(20).apply(lambda x: ((x - np.mean(x))**4).mean() / (np.std(x)+1e-9)**4).shift(1)

        # 24. Jour de la semaine (one-hot) — pas besoin de shift (info connue à t)
        if 'dow' in self.features:
            df['dow'] = pd.to_datetime(df.index).weekday  # 0=Monday, ..., 4=Friday

        # 25. Holiday/Weekend proximity (pas besoin de shift non plus)
        if 'days_to_monday' in self.features:
            df['days_to_monday'] = (7 - pd.to_datetime(df.index).weekday) % 7

        df = df.dropna()
        return df
        


    def _process_binance(self, df):
        # Ajoute ici tes features custom Binance
        if 'returns_1h' in self.features:
            df['returns_1h'] = df['close'].pct_change()
        if 'rsi_14' in self.features:
            import ta
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        # ... autres features
        df = df.dropna()
        return df
