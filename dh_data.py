"""
Module docstring - TBA
"""

import numpy as np

def generate_GBM_data(N, sigma, d, n, K, T, S, option_type='call'):
    """
    Simulates geometric Brownian motion (GBM) price paths and
    computes European option payoffs.

    Parameters
    ----------
    N : int
        Number of simulated price paths (sample size).
    sigma : float
        Volatility of the underlying asset.
    d : int
        Number of assets (dimensions).
    n : int
        Number of hedging time steps (discrete intervals in [0, T]).
    K : float
        Strike price of the European option.
    T : float
        Time to maturity (in years).
    S : float
        Initial asset price.
    option_type : str, optional (default='call')
        Type of European option. Choose 'call' or 'put'.

    Returns
    -------
    price_path : ndarray of shape (N, n+1, d)
        Simulated GBM price paths over time for each asset.
    payoff : ndarray of shape (N, d)
        Option payoff at maturity for each path.
    """
    
    time_grid = np.linspace(0, T, n+1) # Discretized time-to-maturity
    dt = T/n                           # Time increment
    
    BM_path_nozero = np.cumsum(
        np.random.normal(loc=0, scale=np.sqrt(dt), size=(N,n,d)),
        axis = 1
    )
    # Add 0 to each path for B_0=0
    BM_path = np.concatenate([np.zeros((N,1,d)), BM_path_nozero], axis = 1)
    
    # Convert BM to a GBM price path: S_t = S exp(sigma B_t - 0.5sigma^2 t)
    price_path = S * np.exp(sigma * BM_path - 0.5 * sigma**2 * time_grid[None, :, None])
    
    # Calculate the European option payoff
    if option_type=='call':
            payoff_func = lambda x: 0.5* (np.abs(x-K)+x-K)
    else:
        payoff_func = lambda x: 0.5 * (np.abs(x - K) + K - x)
    payoff = payoff_func(price_path[:,-1,:])
    
    return price_path, payoff