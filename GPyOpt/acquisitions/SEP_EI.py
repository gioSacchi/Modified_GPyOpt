from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from ..util.general import reverse_gpyopt_encoding
import numpy as np
from scipy.stats import norm
    
class AcquisitionSepEI(AcquisitionBase):
    
    """
    Expected improvement acquisition function for separated bayesian optimization.

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param objective_function: function to optimize, takes input (X, model(X))
    :param distance_function: function to compute distance between points
    :param current_point: 'starting point', used to compute distance
    :param desired_output: desired output of the objective function
    :param lam: lambda parameter of the acquisition function
    :param cost_withGradients: function that provides the evaluation cost and its gradients
    :param jitter: jitter to avoid numerical issues
    :param g_inv: inverse of the link function g (use is not implemented yet)
    :param normalizer: normalizer of the model outputs

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = False

    
    def __init__(self, model, space, optimizer, objective_function, distance_function, desired_output, lam, cost_withGradients=None, jitter = 0, g_inv = None, **kwargs):
        self.optimizer = optimizer
        super(AcquisitionSepEI, self).__init__(model, space, optimizer)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients 
        self.jitter = jitter
        self.objective_function = objective_function
        self.distance_function = distance_function
        self.desired_output = desired_output
        self.lam = lam
        self.g_inv = g_inv
        self.neg_tol = 1e-10
        self.normalizer = None


    def _compute_optimal(self, one_hot=False):
        model_values = self.model.predict(self.model.model.X)[0]
        if self.normalizer is not None:
            model_values = self.normalizer.denormalize(model_values)
        # TODO: check if this is correct, why not self.model.Y ?
        # print('checking if this is correct, why not self.model.Y ?')
        # print('self.model.Y', self.model.model.Y, self.model.model.Y.shape)
        # print('model_values', model_values, model_values.shape)
        # print('log-diff', np.log(self.model.model.Y - model_values))
        if not one_hot:
            X = self.model.model.X
        else:
            # The x values have been transfomed with GPyOpt one-hot encoding
            X = reverse_gpyopt_encoding(self.model.model.X, self.space.space)
        objective_values = self.objective_function(X, model_values)
        optimal_value = np.min(objective_values)
        return optimal_value

    def _compute_acq(self, x):
        # Expected improvement acquisition function
        try:
            dist = self.distance_function(x).reshape(-1, 1)
            opt = self._compute_optimal()
        except:
            # check if dimensions indicate one-hot encoding of x
            size_if_one_hot = len(self.space.get_bounds())
            if x.shape[1] == size_if_one_hot:
                # The x values have been transfomed with GPyOpt one-hot encoding
                x_reversed = reverse_gpyopt_encoding(x, self.space.space)
                dist = self.distance_function(x_reversed).reshape(-1, 1)
                opt = self._compute_optimal(one_hot=True)
            else:
                raise ValueError('x has wrong dimensions for distance function')
        
        f_acqu_x = np.zeros(dist.shape)
        idx = np.where(dist < opt)[0]
        if len(idx) == 0:
            return f_acqu_x
        
        quat = (opt - self.jitter - dist[idx]) / self.lam
        if self.g_inv is None: # standard Watcher et al. CF formulation
            # ingore warning about invalid value encountered in sqrt
            with np.errstate(invalid='ignore'):
                UB = self.desired_output + np.sqrt(quat)
                LB = self.desired_output - np.sqrt(quat)
        else:
            pass # TODO: implement this using g_inv
            
        mu, std = self.model.predict(x[idx]) # used to be clip, byt GPy does already clip
        
        if self.normalizer is not None:
            mu = self.normalizer.denormalize(mu)
            std = self.normalizer.denormalize_std(std)

        mu = mu.reshape(-1, 1)
        std = std.reshape(-1, 1)

        f1 = opt - self.jitter - dist[idx] - self.lam*((self.desired_output - mu)**2 + std**2)
        f2 = self.lam*std*(mu + UB - 2*self.desired_output)
        f3 = self.lam*std*(2*self.desired_output - mu - LB)

        arg_UB = (UB - mu) / std
        arg_LB = (LB - mu) / std

        f_acqu_x[idx] = f1*(norm.cdf(arg_UB)-norm.cdf(arg_LB)) + f2*norm.pdf(arg_UB) + f3*norm.pdf(arg_LB)

        if np.any(f_acqu_x < 0):
            # check if the problem is due to numerical error
            if np.all(np.abs(f_acqu_x[f_acqu_x<0]) < self.neg_tol):
                f_acqu_x[f_acqu_x<0] = 0
                print('Numerical error in EI, setting to 0')
            else:
                raise ValueError('ei must be non-negative')
                                    
        return f_acqu_x
    
    def _compute_acq_withGradients(self, x):
        
        # --- DEFINE YOUR AQUISITION (TO BE MAXIMIZED) AND ITS GRADIENT HERE HERE
        #
        # Compute here the value of the new acquisition function. Remember that x is a 2D  numpy array  
        # with a point in the domanin in each row. f_acqu_x should be a column vector containing the 
        # values of the acquisition at x. df_acqu_x contains is each row the values of the gradient of the
        # acquisition at each point of x.
        #
        # NOTE: this function is optional. If note available the gradients will be approxiamted numerically.
        
        return f_acqu_x, df_acqu_x