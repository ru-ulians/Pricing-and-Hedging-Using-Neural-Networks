"""
File docstring later - ADD
"""
import numpy as np
from scipy.stats import norm

def set_global_seed(seed=42, deterministic=False):
    """
    Sets random seeds for Python, NumPy, and TensorFlow to ensure reproducibility.

    Parameters:
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


def BlackScholesPremium(T, S, K, sigma, r=0, alldata=False, option_type='call'):
    """
    Calculates the Black-Scholes price of a European call or put option, with optional return of Greeks.

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
    alldata : bool, optional
        If True, returns a dictionary containing price and Greeks.
    option_type : str, optional
        'call' or 'put' (default is 'call').

    Returns
    -------
    float or dict
        Option price, or a dictionary with price and Greeks if `alldata=True`.

    Raises
    ------
    ValueError
        If `option_type` is not 'call' or 'put'.
    """

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
    
    if alldata==True:
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
    
