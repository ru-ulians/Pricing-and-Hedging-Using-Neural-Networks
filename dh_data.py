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


def generate_jump_diffusion_data(jump_int, mu_jump, sigma_jump, N, sigma, d, n, K, T, S, option_type='call'):
     """
    Simulates asset price paths using a jump-diffusion model and computes European option payoffs at maturity.

    The asset follows a geometric Brownian motion (GBM) with an additional compound Poisson jump process,
    where jump magnitudes are normally distributed. Jump effects are applied independently at each time step.

    Parameters
    ----------
    jump_int : float
        Jump intensity (expected number of jumps per unit time).
    mu_jump : float
        Mean of the jump size distribution.
    sigma_jump : float
        Standard deviation of the jump size distribution.
    N : int
        Number of simulated paths.
    sigma : float
        Volatility of the Brownian motion component.
    d : int
        Number of assets (dimensions).
    n : int
        Number of time steps.
    K : float
        Strike price of the option.
    T : float
        Time to maturity.
    S : float
        Initial asset price.
    option_type : str, optional (default='call')
        Type of European option: 'call' or 'put'.

    Returns
    -------
    price_paths : ndarray of shape (N, n+1, d)
        Simulated asset price paths.
    payoff : ndarray of shape (N, d)
        Option payoffs at maturity.
    """
     time_grid = np.linspace(0,T,n+1) # Discretized time-to-maturity
     dt = T/n                         # Time increment

     # Brownian part
     BM_path_nozero = np.cumsum(
        np.random.normal(loc=0, scale=np.sqrt(dt), size=(N,n,d)),
        axis = 1
     )
     # Add 0 to each path for B_0=0
     BM_path = np.concatenate([np.zeros((N,1,d)), BM_path_nozero], axis = 1)

     # Poisson jumps
     number_jumps = np.random.poisson(jump_int*dt, size = (N, n, d))
     jump_sizes = np.random.normal(loc = mu_jump, scale = sigma_jump, size = (N, n, d))
     jumps = number_jumps * jump_sizes
     jumps = np.concatenate([np.zeros([N, 1, d]),jumps],axis=1)

     #Combine
     log_price = np.log(S)+sigma*BM_path - 0.5*sigma**2*time_grid[None,:,None]+ jumps
     price_paths = np.exp(log_price)

     # Calculate the European option payoff
     if option_type=='call':
          payoff_func = lambda x: 0.5* (np.abs(x-K)+x-K)
     else:
         payoff_func = lambda x: 0.5 * (np.abs(x - K) + K - x)
     payoff = payoff_func(price_paths[:,-1,:])

     return price_paths, payoff