import numpy as np
import tensorflow as tf
from edward.models import Normal, Bernoulli


###################################
#
# User defined exceptions
#
###################################

class DimensionTypeError(TypeError):
    def __init__(self, dims):
        print("""Incorrect type for initialising dimensions. 
        Input of type {} but need input list""".format(type(dims)))

class WeightTypeError(TypeError):
    def __init__(self, weights):
        print("""Incorrect type for initialising weights. 
        Input of type {} but need input list""".format(type(weights)))

class BiasTypeError(TypeError):
    def __init__(self, bias):
        print("""Incorrect type for initialising bias. 
        Input of type {} but need input list""".format(type(bias)))        

class DistNotImplementedError(NotImplementedError):
    def __init__(self, dist):
        print("{} distribution not implemented".format(dist))

class InvalidHyperparamsError(ValueError):
    def __init__(self, hyperparams):
        print("Invalid keys in hyperparameters, keys supplied are:")
        for key in hyperparams.keys():
            print(key)
        print("Keys should be \"mean\" and \"var\" " )

class InvalidActivationError(ValueError):
    def __init__(self, activation):
        print("Invalid activation name \"{}\" supplied")
    



def initialise_params(dims, dist = "normal", hyperparams = None):
    """
    Description:
    Will in initialise parameters based of the Bayesian NN 
    Keyword arguments:
    dims (list(ints)) indicate the number of hidden units
        in each hidden layer
    weights (list(tf.tensor)) Edward model variables 
        that hold the weights
    bias (list(tf.tensor)) Edward model variables 
        that hold the bias'
    dist (string) an optional input; a string that indicates the type of 
        distribution we want
    hyperparams (dict) an optional input; will hold hyperparameters
        of how to initialise the the weights/bias of each layer
        Eg. for a normal distribution {'mean':0, 'var':1}
    seed (bool) whether we want to seed the RNG
    Returns:
        (list(tf.tensor)) of layer weights
        (list(tf.tensor)) of layer bias vectors
    """
    #check all our inputs are correct format
    if(not isinstance(dims, list)):
        raise DimensionTypeError(dims)
    #if we need to se the seed
    #currently just use a constant seed value
    if(seed):
        ed.util.set_seed(0)
    #now initialise the params as specified
    if(dist == "normal"):
        weights, bias = _initialise_normal(dims, hyperparams)
    else:
        raise DistributionNotImplementedError(dist)
    
    return weights, bias
    


def simple_feed_forward(X, weights, bias, activation):
    """
    feed_forward()
    Description:
    Will forward pass input data through the layers
    Args:
        X (tf.tensor) Input data as either an matrix or as a column 
            vector weights (tf.tensor)  list of weights for each layer
        activation (list(str)) list of strings indicating how activation 
            is done at each layer
        weights (list(tf.tensor)) list of weights for each layer
        bias (list(tf.tensor)) list of weights for each layer
        activation (list(str)) list of string indicating the activation 
            type for each layer
    Returns:
        (tf.tensor) output of network
    """
    A = X
    for l in range(1, len(weights)):  
        Z = tf.matmul(weights[l], x) + bias[l]
        #apply non-linear activation function
        if(activation[l] == "tanh"):
            A = tf.tanh(Z)
        elif(activation[l] == "relu"):
            A = tf.nn.relu(Z)
        elif(activation[l] == "sigmoid"):
            A = tf.sigmoid(Z)
        #if no activation is applied
        elif(activation[l] == "none"):
            A = Z
        #otherwise an invalid activation function has been supplied
        else:
            raise InvalidActivationError(activation[l])
    return A



def _initialise_normal(dims, hyperparams):
    """
    initialise_normal()
    
    Description:
    Will in initialise parameters for Bayesian NN with Normal distribution
    Args:
        dims = a list of ints that indicate the number of hidden units 
        in each hidden layer
        hyperparams = an optional input; dict that will hold hyperparameters 
        of how to initialise the the weights/bias of each layer.
        Eg. for a normal distribution {'mean':0, 'var':1}
    """
    #check that the hyperparameters are correct
    if(hyperparams == None):
        hyperparams = {"mean":0, "var":1}
    if('mean' not in hyperparams.keys()) or ('var' not in hyperparams.keys()):
        raise InvalidHyperparamsError(hyperparams)

    weights = []
    bias = []
    mean = np.float(hyperparams['mean'])
    var = np.float(hyperparams['var'])
    
    for ii in range(1, len(dims)):
        weights.append(Normal(loc=tf.ones([dims[ii], dims[ii-1]]) * mean,
                               scale=tf.ones([dims[ii], dims[ii-1]])*var))
        bias.append(Normal(loc=tf.ones([dims[ii], dims[ii-1]]) * mean,
                               scale=tf.ones([dims[ii], 1])*var))
    return weights, bias
