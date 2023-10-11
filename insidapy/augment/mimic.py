import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


##########################################
##########################################
### Fit and augment data
##########################################
##########################################
class fit_and_augment():
    """
    This class takes observed data and a suggested ODE model as input. It then uses an indicated method to 
    estimate the parameters of the given model. Last, it uses the identified model to create new data points.
    This procedure can be used to augment the small sample data set that might be available from measurements.

    Args:
        y (array, required): Observed data of shape [n,m] where n is the number of time points and m is the number of species.
        t (array, required): Time points of shape [n,].
        nparams (int, required): Number of parameters to estimate in the ODE model.
        model (callable, required): Callable function that represents the ODE model. Should be in the form of f(t,y,params).
        parameter_bounds (array, required): Bounds for the parameters of shape [nparams,2]. 

    Stores:
        :y (array): Observed data.
        :t (array): Time points.
        :nparams (int): Number of parameters to estimate. 
        :model (callable): Callable function that represents the ODE model.
        :parameter_bounds (array): Bounds for the parameters.
    """
    
    # ----------------------------------------
    def __init__(self, 
                 y:np.ndarray, 
                 t:np.ndarray,
                 nparams:int,
                 model:callable,
                 parameter_bounds:np.ndarray):
        """Initialize function.
        """

        # Store inputs
        self.y = y
        self.t = t
        self.nparams = nparams
        self.model = model
        self.parameter_bounds = parameter_bounds

        # Set flags to check that the user executed the required functions
        self.flag_fit_done = False

    # ----------------------------------------
    def fit(self, 
            method:str='Nelder-Mead', 
            objective:str='RMSE', 
            num_random_search_steps:int=5):
        """Fits the model to the observed data. A variety of existing optimizer methods can be used.

        Args:
            method (str, optional): Optimization method for the parameter estimation.
                Default is currently set to the scipy method 'Nelder-Mead'.
            objective (str, optional): Residual definition. Defaults to 'RMSE' (root mean squared error).
            num_random_search_steps (int, optional): Number of random search steps to find a good starting point for the optimization. Defaults to 5.

        Stores:
            :y0 (array): Initial condition for the ODE solver. The first data point is assumed to be the initial condition.
            :fittingresult (object): Result of the optimization routine (for the scipy implementations, see scipy.optimize.minimize)
            :xopt (array): Estimated parameters.
            :fopt (float): Objective function value at the optimum.
        """
        
        # Define available scipy-methods for minimization
        scipy_methods = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B','TNC', 
                        'COBYLA','SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']
        
        # Use the first data point as initial condition and allocate the time span
        self.y0 = self.y[0,:]
        tspan = self.t

        # Define objective function
        if objective == 'RMSE':
            def objfunc(decisionvars):
                # Solve ODE with parameters
                y_est = self.solve_ode(fun=self.model, tspan=tspan, params=decisionvars)
                # Get residuals
                residuals = self.rmse_(y_est, self.y)
                return residuals
        
        # Get useful starting point by conducting a random search
        _, startingpoint = self.random_search(mdl=objfunc, n_params=self.nparams, bounds_rs=self.parameter_bounds, iter_rs=num_random_search_steps)
        
        # Solve the optimization problem with the indicated method
        if method in scipy_methods:
            self.fittingresult = minimize(objfunc, startingpoint, method=method, bounds=None)
            self.xopt = self.fittingresult.x
            self.fopt = self.fittingresult.fun

        # Set the flag to indicate that the fitting was done
        self.flag_fit_done = True

    # ----------------------------------------
    def predict(self, plotting:bool=False):
        """Uses the estimated parameters to simulate the state profiles (solve the provided ODE model).

        Args:
            plotting (bool, optional): If True, the fit is plotted. Defaults to False.
        """
        
        # Check the fitting was done
        assert self.flag_fit_done, '[-] Please run the fit() function first.'

        # Apply the estimated parameters to the model
        self.apply_estimated_parameters()

        # Plot the fit if required
        if plotting:
            self.plot_fit()

    # ----------------------------------------
    def fit_predict(self, method:str='Nelder-Mead', objective:str='RMSE', num_random_search_steps:int=5, plotting:bool=False):
        """Convenience function to fit and predict in one step.

        Args:
            method (str, optional): Optimization method for the parameter estimation.
                Default is currently set to the scipy method 'Nelder-Mead'.
            objective (str, optional): Residual definition. Defaults to 'RMSE' (root mean squared error).
            num_random_search_steps (int, optional): Number of random search steps to find a good starting point for the optimization. Defaults to 5.
            plotting (bool, optional): If True, the fit is plotted. Defaults to False.
        """
        
        # Fit the model
        self.fit(method=method, objective=objective, num_random_search_steps=num_random_search_steps)

        # Predict the state profiles
        self.predict(plotting=plotting)

    # ----------------------------------------
    def objective_rmse(self, mdl, params=None):
        
        # Solve ODE with parameters
        y_est = self.solve_ode(fun=mdl, y0=y0, tspan=tspan, params=params)
        
        # Get residuals
        residuals = self.rmse_(y_est, y)
        return residuals
    
    # ----------------------------------------
    def random_search(self, mdl=None, n_params=None, bounds_rs=None, iter_rs=None):
        '''This function is a naive optimization routine that randomly samples the
        allowed space and returns the best value. This is used to find a good starting point for the optimization later on.

        Args:
            f (function, required): callable function to optimize.
            n_p (int, required): number of points to sample.
            bounds_rs (np.array, required): bounds of the search space.
            iter_rs (int, required): number of iterations.
        '''

        # arrays to store sampled points
        localx = np.zeros((n_params,iter_rs))        # points sampled
        localval = np.zeros((iter_rs))          # function values sampled
        
        # bounds
        bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
        bounds_bias = bounds_rs[:,0]

        for sample_i in range(iter_rs):
            x_trial = np.random.uniform(0, 1, n_params)*bounds_range + bounds_bias # sampling
            localx[:,sample_i] = x_trial
            localval[sample_i] = mdl(x_trial) # f
       
        # choosing the best
        minindex = np.argmin(localval)
        f_b = localval[minindex]
        x_b = localx[:,minindex]

        return f_b,x_b

    # ----------------------------------------
    def solve_ode(self, fun, tspan, params):
        # Solve ODE with the params
        sol = solve_ivp(fun=fun,y0=self.y0,t_span=[tspan[0],tspan[-1]],t_eval=tspan, args=(params,))
        y_est = sol.y.T
        return y_est

    # ----------------------------------------
    def rmse_(self, y_est, y_obs):
        return np.sqrt(mean_squared_error(y_obs, y_est))

    # ----------------------------------------
    def apply_estimated_parameters(self):
        self.y_fit = self.solve_ode(fun=self.model, tspan=self.t, params=self.xopt)

    # ----------------------------------------
    def plot_fit(self):
        plt.figure()
        for i in range(self.y.shape[1]):
            lablobs = 'observed' if i == 0 else '_Hidden'   # shows only last entry in legend
            lablest = 'estimated' if i == 0 else '_Hidden'  # shows only last entry in legend
            plt.plot(self.t, self.y[:,i], 'o', label=lablobs)
            plt.plot(self.t, self.y_fit[:,i], 'k--', label=lablest)
        plt.xlabel('Time')
        plt.ylabel('Estimated states')
        plt.legend(frameon=False, loc='best')
        # Change color of the legend entries to not confuse the reader
        ax = plt.gca()
        leg = ax.get_legend()
        for L in leg.legendHandles:
            L.set_color('black')
        plt.tight_layout()



