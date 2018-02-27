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
        print("""Incorrect type for initialising dimensions. Input of type {} but need input
        list""".format(type(dims)))

class WeightTypeError(TypeError):
    def __init__(self, weights):
        print("""Incorrect type for initialising weights. Input of type {} but need input
        list""".format(type(weights)))

class BiasTypeError(TypeError):
    def __init__(self, bias):
        print("""Incorrect type for initialising bias. Input of type {} but need input
        list""".format(type(bias)))        

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
    initialise_params()
    
    Description:
    Will in initialise parameters based of the Bayesian NN 
    Keyword arguments:
    dims = a list of ints that indicate the number of hidden units in each hidden layer
    weights = a list of Edward model variables that hold the weights
    bias = a list of Edward model variables that hold the bias'
    dist = an optional input; a string that indicates the type of distribution we want
    hyperparams = an optional input; dict that will hold hyperparameters of how to initialise the 
		  the weights/bias of each layer.
		  Eg. for a normal distribution {'mean':0, 'var':1}
"""
    #check all our inputs are correct format
    if(not isinstance(dims, list)):
	raise DimensionTypeError(dims)
    
    #now initialise the params as specified
    if(dist == "normal"):
	weights, bias = _initialise_normal(dims, hyperparams)
    else:
	raise DistributionNotImplementedError(dist)
    
    return weights, bias
    


def simple_feed_forward(x, weights, bias, activation):
    """
    feed_forward()
    Description:
    Will forward pass input data through the layers
    Args:
	x (tf.tensor) Input data as either an matrix or as a column vector
	weights (tf.tensor)  list of weights for each layer
	activation (list) list of strings indicating how activation is done at each layer
    Returns:
	(tf.tensor) output of network
    """
    for l in range(1, len(weights)):
	#for the first pass will initialise output of the individual layer as the input data
	if l = 0:
	    A = X
	Z = tf.matmul(weights[0], x) + bias[0]
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
    Keyword arguments:
    dims = a list of ints that indicate the number of hidden units in each hidden layer
    hyperparams = an optional input; dict that will hold hyperparameters of how to initialise the 
		  the weights/bias of each layer.
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
