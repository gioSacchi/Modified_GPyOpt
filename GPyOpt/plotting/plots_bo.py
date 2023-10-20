# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab
from ..util.general import Normalize_SEP


def plot_acquisition(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample,
                     filename=None, label_x=None, label_y=None, color_by_step=True):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:
        # X = np.arange(bounds[0][0], bounds[0][1], 0.001)
        # X = X.reshape(len(X),1)
        # acqu = acquisition_function(X)
        # acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))) # normalize acquisition
        # m, v = model.predict(X.reshape(len(X),1))
        # plt.ioff()
        # plt.figure(figsize=(10,5))
        # plt.subplot(2, 1, 1)
        # plt.plot(X, m, 'b-', label=u'Posterior mean',lw=2)
        # plt.fill(np.concatenate([X, X[::-1]]), \
        #         np.concatenate([m - 1.9600 * np.sqrt(v),
        #                     (m + 1.9600 * np.sqrt(v))[::-1]]), \
        #         alpha=.5, fc='b', ec='None', label='95% C. I.')
        # plt.plot(X, m-1.96*np.sqrt(v), 'b-', alpha = 0.5)
        # plt.plot(X, m+1.96*np.sqrt(v), 'b-', alpha=0.5)
        # plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
        # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        # plt.title('Model and observations')
        # plt.ylabel('Y')
        # plt.xlabel('X')
        # plt.legend(loc='upper left')
        # plt.xlim(*bounds)
        # grid(True)
        # plt.subplot(2, 1, 2)
        # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        # plt.plot(X,acqu_normalized, 'r-',lw=2)
        # plt.xlabel('X')
        # plt.ylabel('Acquisition value')
        # plt.title('Acquisition function')
        # grid(True)
        # plt.xlim(*bounds)

        if not label_x:
            label_x = 'x'

        if not label_y:
            label_y = 'f(x)'

        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(x_grid)


        model.plot_density(bounds[0], alpha=.5)

        plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
        plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
        plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

        plt.plot(Xdata, Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

        plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')


        if filename!=None:
            savefig(filename)
        else:
            plt.show()

    if input_dim == 2:

        if not label_x:
            label_x = 'X1'

        if not label_y:
            label_y = 'X2'

        n = Xdata.shape[0]
        colors = np.linspace(0, 1, n)
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=1)
        points_var_color = lambda X: plt.scatter(
            X[:,0], X[:,1], c=colors, label=u'Observations', cmap=cmap, norm=norm)
        points_one_color = lambda X: plt.plot(
            X[:,0], X[:,1], 'r.', markersize=10, label=u'Observations')
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.ylabel(label_y)
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'m.', markersize=10)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        if filename!=None:
            savefig(filename)
        else:
            plt.show()


def plot_convergence(Xdata, best_Y, distance=None, filename=None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = Xdata.shape[0]
    if distance==None:
        aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
        distances = np.sqrt(aux.sum(axis=1))
    else:
        distances = []
        for i in range(n-1):
            distances.append(distance(Xdata[i,:], Xdata[i+1,:])[0,0])

    ## Distances between consecutive x's
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n-1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)),best_Y,'-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    grid(True)

    if filename!=None:
        savefig(filename)
    else:
        plt.show()

def current_plot(bounds, input_dim, model, Xdata, Ydata, next_x, desired, obj_func, inner_function, acquisition_func):
    if input_dim == 1:
        x = np.arange(bounds[0][0], bounds[0][1], 0.01).reshape(-1, 1)
        y = inner_function(x)
        obj_val = obj_func(x, y)
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=.5)

        norm = acquisition_func.normalizer
        mu, std = model.predict(x)
        mu = norm.denormalize(mu).reshape(-1, 1)
        std = norm.denormalize_std(std).reshape(-1, 1)

        # plot objective function
        plt.subplot(1, 3, 1)
        plt.plot(x, obj_val, 'b-', label=u'Objective function')

        # compute upper and lower of surrogate function
        upper = mu + 1.96 * std
        lower = mu - 1.96 * std
        # upper and lower bounds of objective function
        obj_desired = obj_func(x, np.ones_like(x)*desired)
        obj_upper = obj_func(x, upper)
        obj_lower = obj_func(x, lower)
        # take into consideration the objective function non monotonicity
        plot_upper = np.maximum(obj_upper, obj_lower)
        plot_lower = np.where(np.logical_and(desired < obj_upper, desired > obj_lower), obj_desired, np.minimum(obj_upper, obj_lower))
        obj_surrogate = obj_func(x, mu)

        # plot surrogate function
        plt.plot(x, obj_surrogate, 'r-', label=u'Surrogate objective function')
        plt.fill_between(x.ravel(), plot_lower.ravel(), plot_upper.ravel(), alpha=.2, fc='b', ec='None', label='95% C. I.')

        # plot samples
        plt.plot(Xdata, Ydata[:, 1], 'kx', mew=3, label=u'Samples')
        # plot optimum
        opt = np.argmin(Ydata[:, 1])
        plt.plot(Xdata[opt], Ydata[opt, 1], 'k*', mew=3, ms=10, label=u'Optimum')

        # plot inner function
        plt.subplot(1, 3, 2)
        plt.plot(x, y, 'g-', label=u'Real inner function')
        plt.fill_between(x.ravel(), upper.ravel(), lower.ravel(), alpha=.2, fc='b', ec='None', label='95% C. I.')
        plt.plot(x, mu, 'r-', label=u'Surrogate inner function')
        plt.plot(Xdata, Ydata[:, 0], 'kx', mew=3, label=u'Samples')

        # plot acquisition function
        plt.subplot(1, 3, 3)
        acq_vals = acquisition_func._compute_acq(x)
        plt.plot(x, acq_vals, 'r-', label=u'Acquisition function')
        plt.axvline(x=next_x, color='k', label=u'Next sample')

        plt.legend(loc='upper left')
        plt.show()