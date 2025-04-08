"""
module docstring
"""

import numpy as np

def generate_GBM_data(N, sigma, d, n, K, T, S):
    """
    Simulates Geometric Brownian Motion (GBM) price paths for an asset price 
    and calculates the corresponding european call option payoff at maturity.

    Parameters
    ----------
    N : int
        Number of simulated price paths.
    sigma : float
        Volatility of the underlying asset.
    d : int
        Number of underlying assets.
    n : int
        Number of hedging intervals.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    S : float
        Initial asset price.

    Returns
    -------
    price_path : np.ndarray of shape (N, n+1, d)
        N simulated GBM paths.
    payoff : np.ndarray of shape (N, d)
        European call option payoff.
    """
    
    time_grid = np.linspace(0, T, n+1) # Discretized time-to-maturity
    dt = T/n                           # Time increment

    BM_path_nozero = np.cumsum(
        np.random.normal(loc=0, scale=np.sqrt(dt), size=(N,n,d)),
        axis = 1
    )
    # Add 0 to each path for B=0
    BM_path = np.concatenate([np.zeros((N,1,d)), BM_path_nozero], axis = 1)

    # Convert BM to a GBM price path: S_t = S exp(B_t - 0.5sigma^2 t)
    price_path = S * np.exp(sigma * BM_path - 0.5 * sigma**2 * time_grid[None, :, None])

    # Calculate the European call option payoff
    payoff_func = lambda x: 0.5* (np.abs(x-K)+x-K)
    payoff = payoff_func(price_path[:,-1,:])

    return price_path, payoff

    
def generate_jump_diffusion_data(jump_intensity, mu_jump, sigma_jump, T, n, N, d, S, sigma, K):
    """
    Simulates asset price paths using a jump-diffusion model and computes option payoffs.

    The model combines geometric Brownian motion with normally distributed jump sizes
    and Poisson-distributed jump counts. It returns simulated price paths and payoffs
    for a European option.

    Parameters
    ----------
    jump_intensity : float
        Intensity of the Poisson process representing jumps per unit time.
    mu_jump : float
        Mean of the jump size distribution.
    sigma_jump : float
        Standard deviation of the jump size distribution.
    T : float
        Time to maturity.
    n : int
        Number of discrete time steps.
    N : int
        Number of simulated paths.
    d : int
        Number of underlying assets.
    S : float
        Initial asset price.
    sigma : float
        Volatility of the Brownian component.
    K : float
        Strike price of the option.

    Returns
    -------
    price_path : ndarray of shape (N, n+1, d)
        Simulated asset price paths under the jump-diffusion process.
    payoff : ndarray of shape (N, d)
        European call option payoff.
    """
    time_grid = np.linspace(0, T, n + 1)
    dt = T / n

    # Brownian part
    path_helper = np.cumsum(np.random.normal(loc=0, scale = np.sqrt(dt),size = (N, n, d)), axis = 1)
    path = np.concatenate([np.zeros([N, 1, d]), path_helper], axis = 1)

    # Poisson Jumps
    number_jumps = np.random.poisson(jump_intensity * dt, size = (N, n+1, d))
    jump_sizes = np.random.normal(loc = mu_jump, scale = sigma_jump, size = (N, n, d))
    jumps = np.cumsum(number_jumps*jump_sizes, axis = 1)
    jumps = np.concatenate([np.zeros([N, 1, d]), jumps], axis = 1)

    log_price = np.log(S) + sigma * path - 0.5 * sigma**2 * time_grid[None,:,None] + jumps
    price_path = np.exp(log_price)

    #Payoff
    payoff_func = lambda x: 0.5 * (np.abs(x - K) + x - K)
    payoff = payoff_func(price_path[:,-1,:])

    return price_path, payoff 
