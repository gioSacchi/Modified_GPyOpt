# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import erfc
import time
from ..core.errors import InvalidConfigError

def compute_integrated_acquisition(acquisition,x):
    '''
    Used to compute the acquisition function when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: acquisition function with GpyOpt model type GP_MCMC.
    :param x: location where the acquisition is evaluated.
    '''

    acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i,:]
        acqu_x += acquisition.acquisition_function(x)

    acqu_x = acqu_x/acquisition.model.num_hmc_samples
    return acqu_x

def compute_integrated_acquisition_withGradients(acquisition,x):
    '''
    Used to compute the acquisition function with gradients when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: acquisition function with GpyOpt model type GP_MCMC.
    :param x: location where the acquisition is evaluated.
    '''

    acqu_x = 0
    d_acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i,:]
        acqu_x_sample, d_acqu_x_sample = acquisition.acquisition_function_withGradients(x)
        acqu_x += acqu_x_sample
        d_acqu_x += d_acqu_x_sample

    acqu_x = acqu_x/acquisition.model.num_hmc_samples
    d_acqu_x = d_acqu_x/acquisition.model.num_hmc_samples

    return acqu_x, d_acqu_x


def best_guess(f,X):
    '''
    Gets the best current guess from a vector.
    :param f: function to evaluate.
    :param X: locations.
    '''
    n = X.shape[0]
    xbest = np.zeros(n)
    for i in range(n):
        ff = f(X[0:(i+1)])
        xbest[i] = ff[np.argmin(ff)]
    return xbest


def samples_multidimensional_uniform(bounds,num_data):
    '''
    Generates a multidimensional grid uniformly distributed.
    :param bounds: tuple defining the box constraints.
    :num_data: number of data points to generate.

    '''
    dim = len(bounds)
    Z_rand = np.zeros(shape=(num_data,dim))
    for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
    return Z_rand


def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x

def get_moments(model,x):
    '''
    Moments (mean and sdev.) of a GP model at x

    '''
    input_dim = model.X.shape[1]
    x = reshape(x,input_dim)
    fmin = min(model.predict(model.X)[0])
    m, v = model.predict(x)
    s = np.sqrt(np.clip(v, 0, np.inf))
    return (m,s, fmin)

def get_d_moments(model,x):
    '''
    Gradients with respect to x of the moments (mean and sdev.) of the GP
    :param model: GPy model.
    :param x: location where the gradients are evaluated.
    '''
    input_dim = model.input_dim
    x = reshape(x,input_dim)
    _, v = model.predict(x)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))
    return (dmdx, dsdx)


def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def best_value(Y,sign=1):
    '''
    Returns a vector whose components i are the minimum (default) or maximum of Y[:i]
    '''
    n = Y.shape[0]
    Y_best = np.ones(n)
    for i in range(n):
        if sign == 1:
            Y_best[i]=Y[:(i+1)].min()
        else:
            Y_best[i]=Y[:(i+1)].max()
    return Y_best

def spawn(f):
    '''
    Function for parallel evaluation of the acquisition function
    '''
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun


def evaluate_function(f,X):
    '''
    Returns the evaluation of a function *f* and the time per evaluation
    '''
    num_data, dim_data = X.shape
    Y_eval = np.zeros((num_data, dim_data))
    Y_time = np.zeros((num_data, 1))
    for i in range(num_data):
        time_zero = time.time()
        Y_eval[i,:] = f(X[i,:])
        Y_time[i,:] = time.time() - time_zero
    return Y_eval, Y_time


def values_to_array(input_values):
    '''
    Transforms a values of int, float and tuples to a column vector numpy array
    '''
    if type(input_values)==tuple:
        values = np.array(input_values).reshape(-1,1)
    elif type(input_values) == np.ndarray:
        values = np.atleast_2d(input_values)
    elif type(input_values)==int or type(input_values)==float or type(np.int64):
        values = np.atleast_2d(np.array(input_values))
    else:
        print('Type to transform not recognized')
    return values


def merge_values(values1,values2):
    '''
    Merges two numpy arrays by calculating all possible combinations of rows
    '''
    array1 = values_to_array(values1)
    array2 = values_to_array(values2)

    if array1.size == 0:
        return array2
    if array2.size == 0:
        return array1

    merged_array = []
    for row_array1 in array1:
        for row_array2 in array2:
            merged_row = np.hstack((row_array1,row_array2))
            merged_array.append(merged_row)
    return np.atleast_2d(merged_array)


def normalize(Y, normalization_type='stats'):
    """Normalize the vector Y using statistics or its range.

    :param Y: Row or column vector that you want to normalize.
    :param normalization_type: String specifying the kind of normalization
    to use. Options are 'stats' to use mean and standard deviation,
    or 'maxmin' to use the range of function values.
    :return Y_normalized: The normalized vector.
    """
    Y = np.asarray(Y, dtype=float)

    if np.max(Y.shape) != Y.size:
        raise NotImplementedError('Only 1-dimensional arrays are supported.')

    # Only normalize with non null sdev (divide by zero). For only one
    # data point both std and ptp return 0.
    if normalization_type == 'stats':
        Y_norm = Y - Y.mean()
        std = Y.std()
        if std > 0:
            Y_norm /= std
    elif normalization_type == 'maxmin':
        Y_norm = Y - Y.min()
        y_range = np.ptp(Y)
        if y_range > 0:
            Y_norm /= y_range
            # A range of [-1, 1] is more natural for a zero-mean GP
            Y_norm = 2 * (Y_norm - 0.5)
    else:
        raise ValueError('Unknown normalization type: {}'.format(normalization_type))

    return Y_norm

class Normalize_SEP():
    """Class to handle normalizations for seprated bayesian optimization."""

    def __init__(self, Y_mean = None, normalization_type='stats', Y_std = None, Y_range = None):
        self.normalization_type = normalization_type
        self.Y_mean = Y_mean
        self.Y_std = Y_std
        self.Y_range = Y_range
    
    def store_stats(self, Y):
        self.Y_mean = Y.mean()
        self.Y_std = Y.std()
        self.Y_range = np.ptp(Y)

    def normalize(self, Y):
        Y = np.asarray(Y, dtype=float)

        if np.max(Y.shape) != Y.size:
            raise NotImplementedError('Only 1-dimensional arrays are supported.')
        
        if self.normalization_type == 'stats':
            Y_norm = Y - Y.mean()
            std = Y.std()
            if std > 0:
                Y_norm /= std
        elif self.normalization_type == 'maxmin':
            Y_norm = Y - Y.min()
            y_range = np.ptp(Y)
            if y_range > 0:
                Y_norm /= y_range
                # A range of [-1, 1] is more natural for a zero-mean GP
                Y_norm = 2 * (Y_norm - 0.5)
        else:
            raise ValueError('Unknown normalization type: {}'.format(self.normalization_type))
        
        return Y_norm

    def normalize_and_store(self, Y):
        self.store_stats(Y)
        return self.normalize(Y)
    
    def denormalize(self, Y_norm):
        Y_norm = np.asarray(Y_norm, dtype=float)

        if np.max(Y_norm.shape) != Y_norm.size:
            raise NotImplementedError('Only 1-dimensional arrays are supported.')

        if self.normalization_type == 'stats':
            Y = Y_norm * self.Y_std + self.Y_mean
        elif self.normalization_type == 'maxmin':
            Y = Y_norm / 2 + 0.5
            Y *= self.Y_range
            Y += self.Y_min
        else:
            raise ValueError('Unknown normalization type: {}'.format(self.normalization_type))
        
        return Y
    
    def denormalize_std(self, Y_norm_std):
        Y_norm_std = np.asarray(Y_norm_std, dtype=float)

        if np.max(Y_norm_std.shape) != Y_norm_std.size:
            raise NotImplementedError('Only 1-dimensional arrays are supported.')

        if self.normalization_type == 'stats':
            Y_std = Y_norm_std * self.Y_std
        elif self.normalization_type == 'maxmin':
            # Not sure about this
            Y_std = Y_norm_std / 2
            Y_std *= self.Y_range
        else:
            raise ValueError('Unknown normalization type: {}'.format(self.normalization_type))
        
        return Y_std
    
def reverse_gpyopt_encoding(points, space):
    """Reverse the one-hot encoding used by GPyOpt.
    
    :param points: The points to be decoded.
    :param space: The space used for the encoding.
    :return: The decoded points.
    """

    new_points = np.empty((points.shape[0], len(space)))
    offset = 0
    for i, feature in enumerate(space):
        if feature.type == 'continuous':
            new_points[:, i] = points[:, (i + offset)]
        elif feature.type == 'discrete':
            new_points[:, i] = points[:, (i + offset)]
        elif feature.type == 'categorical':
            # domain size of the categorical variable
            cat_size = len(feature.domain)
            # one-hot encoding
            one_hot = points[:, (i + offset):(i + offset + cat_size)]
            # reverse, index of the 1
            new_points[:, i] = np.argmax(one_hot, axis=1)
            # update offset
            offset += cat_size - 1
        else:
            raise InvalidConfigError('Unknown feature type {}'.format(feature.type))
    return new_points
