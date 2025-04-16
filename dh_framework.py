"""
Deep Hedging Model Model.

This module defines the neural network architecture used in the deep hedging framework,
including:
- A utility function to construct per-step hedging networks
- A custom Keras layer that computes portfolio evolution and pricing
- A model builder that composes the full deep hedging model

These tools support the training of end-to-end hedging strategies under
risk measures (e.g., MSE, CVaR, entropic loss).

Dependencies
------------
- TensorFlow
"""

import tensorflow as tf

def create_networks(d, hidden_nodes, L, n):
    """
    Builds a list of neural networks used in the deep hedging framework.
    Each network corresponds to a single time step and maps the current market state to a hedging position.
    The architecture is a fully connected feedforward network with `L` layers, `hidden_nodes` neurons,
    and a final output of size `d`.

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
    Hedging_Networks : list of tf.keras.Model
        A list of length `n`, where each model maps R^d → R^d.
        These networks represent the hedging strategy at each time step.
    """
    Hedging_Networks = []
    
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
                    name = srt(j)+'step'+srt(i)+'layer'
                )
                outputs = layer(x)
                network = tf.keras.Model(inputs=inputs, outputs=outputs)
                Hedging_Networks.append(network)
    return Hedging_Networks


class DeepHedgingLayer(Layer):
    """
    Custom Keras layer implementing the deep hedging framework for a number of discrete hedging steps.
    
    At each time step t_k, the network F_k approximates the trading strategy delta_k,
    mapping from the observed price S_k to the hedging position.
    The resulting portfolio \Pi accumulates gains from trading and adds a learned initial premium.

    Parameters
    ----------
    Hedging_Networks : list of tf.keras.Model
        A list of neural networks F_k, one for each hedging step.
    Premium_Network : tf.keras.layers.Layer
        A single-layer model that outputs the initial premium \Pi_0.
    n : int
        Number of discrete hedging steps.

    Returns
    -------
    outputs : tf.Tensor of shape (batch_size, d)
        The final hedged portfolio value, computed as:
            \Pi_T = \Pi_0 + \sum_k delta_k * (S_{k+1} - S_k)
        where delta_k is learned via F_k and \Pi_0 is output by Network0.
    """
    def __init__(self, Hedging_Networks, Premium_Network, n, **kwargs):
        super(DeepHedgingLayer, self).__init__(**kwargs)
        self.Hedging_Networks = Hedging_Networks
        self.Premium_Network = Premium_Network
        self.n = n

    def call(self, price):
        """
        A forward pass through the deep hedging layer.
        """
        price_difference = price[:,1:,:] - price[:, :-1,:]          # S_k - S_{k-1}
        hedge = tf.zeros_like(price[:,0,:])                         # \Pi_0 = 0
        premium = self.Premium_Network(tf.ones_like(price[:,0,:]))

        for j in range(self.n):
            strategy = self.Hedging_Networks[j](price[:,j,:])       # delta_k = F_k(S_k)
            hedge += strategy * price_difference[:,j,:]             # \Pi += delta_k * (S_k-S_{k-1})
        
        outputs = premium + hedge                                   # Final portfolio value
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
    Premium_Network : tf.keras.layers.Layer
        The learnable dense layer representing the initial premium.
    Hedging_Networks : list of tf.keras.Model
        List of hedging policy networks, one per time step.
    """
    # List of networks for each time step
    Hedging_Networks = create_networks(
        d=d,
        hidden_nodes=hidden_nodes,
        L=L,
        n=n
    )
    # Initial premium layer
    Premium_Network = tf.keras.layers.Dense(d, use_bias=False)

    # Input: price paths of shape: (Nsample, n+1, d)
    price = tf.keras.Input(shape=(n+1,d))

    # Output : \Pi + PnL
    outputs = DeepHedgingLayer(
        Hedging_Networks=Hedging_Networks,
        Premium_Network=Premium_Network,
        n=n
    )(price)

    model = tf.keras.Model(inputs=price, outputs=outputs)
    return model, Premium_Network, Hedging_Networks