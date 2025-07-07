# performance.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_returns(series):
    """
    Compute periodic returns from a price or equity series.

    :param series: array-like of prices or portfolio values
    :return: pd.Series of periodic returns
    """
    #print(series)
    s = pd.Series(series)
    returns = s.pct_change().fillna(0)
    return returns


def compute_alpha_beta(R_strategy, R_market):
    """
    Estimate CAPM alpha and beta for strategy returns versus market returns.

    :param R_strategy: pd.Series or array-like of strategy returns
    :param R_market: pd.Series or array-like of market returns
    :return: tuple (alpha, beta, residuals)
    """
    # Convert to aligned pandas Series
    r_strat = pd.Series(R_strategy).reset_index(drop=True)
    r_mkt = pd.Series(R_market).reset_index(drop=True)

    # Align lengths
    n = min(len(r_strat), len(r_mkt))
    r_strat = r_strat.iloc[:n]
    r_mkt = r_mkt.iloc[:n]

    # Prepare regression inputs
    X = r_mkt.values.reshape(-1, 1)
    y = r_strat.values

    # Fit linear model for CAPM
    model = LinearRegression().fit(X, y)
    alpha = model.intercept_
    beta = model.coef_[0]

    # Compute residuals (epsilon)
    epsilon = y - model.predict(X)

    return alpha, beta, epsilon