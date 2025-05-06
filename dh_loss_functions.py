"""
Custom loss functions for risk-sensitive deep hedging models.

This module defines differentiable loss functions commonly used in financial
risk management and pricing, including:
- Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR)
- Entropic Risk Measure

These losses can be used with TensorFlow models to train hedging strategies 
that align with different risk preferences.
"""

import tensorflow as tf
import tensorflow_probability as tfp

def VaR(alpha=0.05):
    """
    Returns a loss function that applies an alpha-weighted symmetric penalty
    on the residuals. This is not the true VaR loss, but a convex surrogate.

    Parameters
    ----------
    alpha : float
        Tail risk sensitivity coefficient.

    Returns
    -------
    function
        TensorFlow loss function: mean(alpha * |y_true - y_pred|)
    """
    def loss(y_true, y_pred):
        res = y_true - y_pred
        return tf.reduce_mean(tf.maximum(alpha * res, (alpha - 1) * res))
    return loss


def CVaR(alpha=0.05):
    """
    Returns a loss function that approximates Conditional Value-at-Risk (CVaR),
    i.e., the average of residuals beyond the (1 - alpha)-quantile.

    Parameters
    ----------
    alpha : float
        Tail probability level (e.g., 0.05 for 95% CVaR)

    Returns
    -------
    function
        TensorFlow loss function: empirical CVaR of |y_true - y_pred|
    """
    def loss(y_true, y_pred):
        res = tf.abs(y_true - y_pred)
        var_alpha = tfp.stats.percentile(res, 100 * (1 - alpha), axis=0, interpolation='linear')
        mask = tf.cast(res >= var_alpha, tf.float32)
        cvar = tf.reduce_sum(res * mask) / (tf.reduce_sum(mask) + 1e-6)
        return cvar
    return loss

def entropic(beta=3):
    """
    Returns a loss function implementing the entropic risk measure, a convex
    and coherent alternative to variance-based risk.

    Parameters
    ----------
    beta : float
        Risk aversion coefficient.

    Returns
    -------
    function
        TensorFlow loss: (1 / beta) * log(mean(exp(beta * |residual|)))
    """
    def loss(y_true, y_pred):
        res = tf.abs(y_true - y_pred)
        scaled = tf.clip_by_value(beta * res, -50.0, 50.0)
        entropic_risk = tfp.math.reduce_logmeanexp(scaled, axis=0) / beta
        return tf.reduce_mean(entropic_risk)
    return loss