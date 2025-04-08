"""
Deep Hedging Model Components.

This module defines the neural network architecture used in the deep hedging framework,
including:

- A function to construct per-step hedging networks
- A custom Keras layer that computes portfolio evolution and pricing
- A model builder that composes the full deep hedging model

Dependencies
------------
- TensorFlow
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

def create_networks(d, hidden_nodes, L, n):
    """
    Builds a list of neural networks used in the deep hedging framework.
    Each network corresponds to a single time step and maps the current market state to a hedging position.
    The architecture is a fully connected feedforward network with `L` layers, `hidden_nodes` neurons, and a final output of size `d`.

    Parameters
    ----------
    d : int
        Dimension of the input and output vectors.
    hidden_nodes : int
        Number of neurons in each hidden layer.
    L : int
        Number of layers in each network.
    n : int
        Number of discrete hedging time steps.
    
    Returns
    -------
    Networks : list of tf.keras.Model
        A list of length `n`, where each model is the hedging strategy at each time step.
    """
    Networks = []
    
    for j in range(n):
        inputs = tf.keras.Input(shape = (d,))
        x = inputs

        for i in range(L):
            if i < L-1:
                layer = tf.keras.layers.Dense(
                    hidden_nodes,
                    activation = 'tanh',
                    trainable = True,
                    kernel_initializer = tf.keras.initializers.RandomNormal(0,1),
                    bias_initializer = 'random_normal',
                    name = str(j)+'step'+str(i)+'layer'
                )
                x = layer(x)
            else :
                layer = tf.keras.layers.Dense(
                    d,
                    activation = 'linear',
                    trainable = True,
                    kernel_initializer = tf.keras.initializers.RandomNormal(0,0.1),
                    bias_initializer = 'random_normal',
                    name = str(j)+'step'+str(i)+'layer'
                )
                outputs = layer(x)
                network = tf.keras.Model(inputs=inputs, outputs=outputs)
                Networks.append(network)
    return Networks


class DeepHedgingLayer(Layer):
    """
    Custom Keras layer implementing the deep hedging framework for a number of discrete hedging steps.
    At each time step t_k, the network F_k approximates the trading strategy delta_k,
    mapping from the observed price S_k to the hedging position.
    The resulting portfolio \Pi accumulates gains from trading and adds a learned initial premium.

    Parameters
    ----------
    Networks : list of keras.Model
        A list of neural networks F_k.
    Network0 :
        A single layer model that outputs the initial premium.
    n : int
        Number of discrete hedging steps.
    
    Returns
    -------
    outputs : tf.Tensor of shape (batch_size, d)
        The final hedged portfolio value.
    """
    def __init__(self, Networks, Network0, n, **kwargs):
        super(DeepHedgingLayer, self).__init__(**kwargs)
        self.Networks = Networks
        self.Network0 = Network0
        self.n = n

    def call(self, price):
        """
        A forward pass through the deep hedging layer.
        """
        price_difference = price[:,1:,:] - price[:, :-1,:]  # S_k - S_{k-1}
        hedge = tf.zeros_like(price[:,0,:])                 # \Pi_0 = 0
        premium = self.Network0(tf.ones_like(price[:,0,:]))

        for j in range(self.n):
            strategy = self.Networks[j](price[:,j,:])       # delta_k = F_k(S_k)
            hedge += strategy * price_difference[:,j,:]     # \Pi += delta_k * (S_k-S_{k-1})
        
        outputs = premium + hedge                           # Final portfolio value
        return outputs
    

def build_dh_model(d, hidden_nodes, L, n):
    """
    Builds and compiles the deep hedging model.

    Parameters
    ----------
    d : int
        Input and output dimension.
    hidden_nodes : int  
        Number of neurons in the hidden layers.
    L : int
        Number of layers in each network.
    n : int
        Number of discrete hedging steps.

    Returns
    -------
    model : tf.keras.Model
        The compiled Keras model mapping price paths to total hedge PnL.
    Network0 : tf.keras.layers.Layer
        The dense layer representing the initial premium.
    Networks : list of tf.keras.Model
        List of hedging networks.
    """
    # List of networks for each time step
    Networks = create_networks(
        d=d,
        hidden_nodes=hidden_nodes,
        L=L,
        n=n
    )
    # Initial premium layer
    Network0 = tf.keras.layers.Dense(d, use_bias=False)

    # Input: price paths of shape: (Nsample, n+1, d)
    price = tf.keras.Input(shape=(n+1,d))

    # Output : \Pi + PnL
    outputs = DeepHedgingLayer(
        Networks=Networks,
        Network0=Network0,
        n=n
    )(price)

    model = tf.keras.Model(inputs=price, outputs=outputs)
    return model, Network0, Networks