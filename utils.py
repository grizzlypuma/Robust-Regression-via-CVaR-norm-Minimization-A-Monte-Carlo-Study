# utils.py

import numpy as np
import pandas as pd

def generate_stock_data(n_points=1000, shock_prob=0.05):
    """
    Generates simulated data for two assets, mimicking a calm market
    with occasional shocks (fat tails).
    """
    mean_normal = [0.0005, 0.0007]
    cov_normal = [[0.0004, 0.0002], [0.0002, 0.0009]]

    mean_shock = [-0.03, -0.04]
    cov_shock = [[0.0050, 0.0040], [0.0040, 0.0060]]
    
    data = []
    for _ in range(n_points):
        if np.random.rand() > shock_prob:
            ret = np.random.multivariate_normal(mean_normal, cov_normal)
        else:
            ret = np.random.multivariate_normal(mean_shock, cov_shock)
        data.append(ret)
        
    df = pd.DataFrame(data, columns=['Asset1', 'Asset2'])
    return df

def calculate_cvar(data, alpha):
    """Calculates CVaR for a given dataset."""
    data = data[~np.isnan(data)]
    var = np.percentile(data, 100 * alpha)
    # Handle the case where all data points are the same
    if len(data[data > var]) == 0:
        return var
    cvar = data[data > var].mean()
    return cvar
