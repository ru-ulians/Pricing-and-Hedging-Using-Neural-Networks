"""
Experiment Functions for Deep Hedging Experiments

This module provides tools for deep hedging testing and analysis, including:

- Plotting utilities for training diagnostics
- Statistical functions for evaluating residuals
- Reproducibility utilities
- Closed-form pricing formulas:
    - Black-Scholes model
    - Merton jump-diffusion model

Functions
---------
- plot_training_curves : Plot training/validation loss curves.
- residual_error_statistics : Summarize residual statistics.
- set_global_seed : Set random seeds for reproducibility.
- BlackScholesPrice : Compute Black-Scholes price and Greeks.
- MertonJumpDiffusionPrice : Compute Merton jump-diffusion option price.

These functions support model training, diagnostics, and benchmarking against analytical solutions.
"""

from scipy.stats import norm
import numpy as np
from scipy.stats import poisson

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_training_curves(histories, titles):
    """
    Plots training and validation loss curves for multiple models.

    Parameters
    ----------
    histories : list of Keras History objects
        Training histories returned by model.fit().
    titles : list of str
        Names of the loss functions/models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, history in enumerate(histories):
        ax = axes[i]
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title(f"{titles[i]} Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def residual_error_statistics(predictions, labels, titles):
    print("\n--- Residual Error Statistics ---")
    for i, pred in enumerate(predictions):
        residuals = pred - labels
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        print(f"{titles[i]} - mean residual: {mean_res: .6f}, std: {std_res: .6f}")

def set_global_seed(seed=42, deterministic=False):
    """
    Sets random seeds for Python, NumPy, and TensorFlow to ensure reproducibility.

    Parameters:
    -----------
    - seed (int): Random seed value.
    - deterministic (bool): If True, enforces deterministic ops in TensorFlow.

    Useful for consistent experimental results across runs.
    """
    import os, random, numpy as np, tensorflow as tf

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'


def BlackScholesPrice(T, S, K, sigma, r=0, greeks=False, option_type='call'):
    """
    Computes the Black-Scholes price of a European call or put option, 
    with optional return of Greeks.

    Parameters
    ----------
    T : float
        Time to maturity (in years).
    S : float
        Current underlying asset price.
    K : float
        Strike price of the option.
    sigma : float
        Volatility of the underlying asset.
    r : float, optional
        Risk-free interest rate (default is 0).
    greeks : bool, optional
        If True, returns a dictionary containing price and Greeks.
    option_type : str, optional
        'call' or 'put' (default is 'call').

    Returns
    -------
    float or dict
        Option price, or a dictionary with price and Greeks if `greeks=True`.

    Raises
    ------
    ValueError
        If `option_type` is not 'call' or 'put'.
    """
    import numpy as np
    from scipy.stats import norm

    d_1 = (np.log(S/K) + (r + sigma**2) * T)/(sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)

    # Greeks common to both call and put
    gamma = norm.pdf(d_1)/(S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d_1)

    # Price and Greeks for the European call option
    if option_type=='call':
        price = S * norm.cdf(d_1) - K * np.exp(-r * T)*norm.cdf(d_2)
        delta = norm.cdf(d_1)
        theta = - (S*norm.pdf(d_1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d_2)
        rho = K*T*np.exp(-r*T)*norm.cdf(d_2)

    # Price and Greeks for the European put option
    elif option_type=='put':
        price = K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
        delta = norm.cdf(d_1) - 1
        theta = - (S*norm.pdf(d_1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d_2)
        rho = -K*T*np.exp(-r*T)*norm.cdf(d_2)

    else:
        raise ValueError(f"Invalid option type: '{option_type}'. Select 'put' or 'call'.")
    
    if greeks==True:
        return {
            'price' : price,
            'delta' : delta,
            'gamma' : gamma,
            'theta' : theta,
            'vega'  : vega,
            'rho'   : rho
        }
    else:
        return price 


def MertonJumpDiffusionPrice(S, K, T, sigma, r, 
                              jump_intensity, mu_jump, sigma_jump, 
                              option_type='call'):
    """
    Computes the Merton jump-diffusion price for a European option
    using a series expansion over Poisson-distributed jump counts.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the Brownian motion.
    jump_intensity : float
        Jump intensity (expected jumps per year).
    mu_jump : float
        Mean of the normal distribution of jump sizes (log-scale).
    sigma_jump : float
        Standard deviation of jump sizes (log-scale).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    price : float
        Option price under the Merton jump-diffusion model.
    """
    # Expected jump size adjustment
    k = np.exp(mu_jump + 0.5 * sigma_jump**2) - 1
    lambda_T = jump_intensity * T
    price = 0.0

    # Sum to 99.9% of the probability mass of the Poisson distribution with mean lambda*T
    N_terms = int(np.ceil(jump_intensity*T + 4*np.sqrt(jump_intensity*T)))

    # Sum over number of jumps n=0 to N_terms
    for n in range(N_terms):
        # Adjusted volatility and drift
        sigma_n = np.sqrt(sigma**2 + n * sigma_jump**2 / T)
        r_n = r - jump_intensity * k + n * mu_jump / T
        weight = poisson.pmf(n, lambda_T)

        # Use your Black-Scholes function
        bs_price = BlackScholesPrice(T, S, K, sigma_n, r=r_n, option_type=option_type)
        price += weight * bs_price

    return price