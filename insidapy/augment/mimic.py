import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import pyDOE
import random


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
        example (str, optional): Name of the example. Default given by 'augment_fit_case_study'.
        name_of_time_vector (str, optional): Name of the time vector. Default is 'time'.
        time_unit (str, optional): Unit of the time. Defaults to 'hours'.
        species_units (list, optional): Units of the different species as a list. If nothing is given, 
            'n.a.' is used for each species.
        
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
                 parameter_bounds:np.ndarray,
                 example:str='augment_fit_case_study',
                 name_of_time_vector:str='time',
                 time_unit:str='h',
                 species_units:list=None):
        """Initialize function.
        """

        # Store inputs
        self.y = y
        self.t = t
        self.nparams = nparams
        self.model = model
        self.parameter_bounds = parameter_bounds
        self.species = [f's{i}' for i in range(self.y.shape[1])]
        self.name_of_time_vector = name_of_time_vector
        self.time_unit = time_unit
        self.example = example

        # Create species units
        if species_units is None:
            self.species_units = ['n.a.' for _ in range(y.shape[1])]
        else:
            self.species_units = species_units

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
        
        # Define objective function
        if objective == 'RMSE':
            def objfunc(decisionvars):
                # Solve ODE with parameters
                y_est = self.solve_ode(fun=self.model, y0=self.y[0,:], tspan=self.t, params=decisionvars)
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
        print('[+] Performt parameter estimation. Stored values under "self.xopt".')

    # ----------------------------------------
    def predict(self,
                show:bool=True,
                save:bool=False, 
                figname:str='figure',
                save_figure_directory:str='./figures', 
                save_figure_exensions:list=['svg','png']):
        """Uses the estimated parameters to simulate the state profiles (solve the provided ODE model).

        Args:
            plotting (bool, optional): If True, the fit is plotted. Defaults to False.
        """
        
        # Check the fitting was done
        assert self.flag_fit_done, '[-] Please run the fit() function first.'

        # Apply the estimated parameters to the model
        self.apply_estimated_parameters()
        print('[+] Performed prediction with identified parameters. Stored under "self.y_fit".')

        # Plot the fit
        self.plot_fit(show, save, figname, save_figure_directory, save_figure_exensions)

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
    def solve_ode(self, fun, y0, tspan, params):
        """Solve the ODE with the given parameters.

        Args:
            fun (callable): Callable function that represents the ODE model.
            y0 (array): Initial condition for the ODE solver. The first data point is assumed to be the initial condition.
            tspan (array): Time points.
            params (array): Parameters.
        """
        # Solve ODE with the params
        sol = solve_ivp(fun=fun, y0=y0, t_span=[tspan[0],tspan[-1]], t_eval=tspan, args=(params,))
        y_est = sol.y.T
        return y_est

    # ----------------------------------------
    def rmse_(self, y_est, y_obs):
        """Returns the root mean squared error (RMSE) of two vectors.

        Args:
            y_est (array): Estimated values (predictions)
            y_obs (array): Observed values

        Returns:
            float: RMSE of the vectors indicated.
        """
        return np.sqrt(mean_squared_error(y_obs, y_est))

    # ----------------------------------------
    def apply_estimated_parameters(self, y0=None, tspan=None, params=None):
        """Uses the estimated parameters to simulate the state profiles (solve the provided ODE model).

        Args:
            y0 (array, optional): Initial conditions of the ODE. Defaults to None. If None, the first data point is assumed to be the initial condition.
            tspan (array, optional): Time points for the ODE integration. Defaults to None. If None, the time points of the observed measurements are used.
            params (array, optional): Parameters for the ODE. Defaults to None. If None, the identified parameters from the fitting are used.
        """
        if y0 is None:
            y0 = self.y[0,:]
        if tspan is None:
            tspan = self.t
        if params is None:
            params = self.xopt
        self.y_fit = self.solve_ode(fun=self.model, y0=y0, tspan=tspan, params=params)

    # ----------------------------------------
    def plot_fit(   self, 
                    show:bool=True,
                    save:bool=False, 
                    figname:str='figure',
                    save_figure_directory:str='./figures', 
                    save_figure_exensions:list=['svg','png']):
        """Plot the fit of the model to the observed data. 
        """
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
        
        # Save figure if required
        self.save_figure(save, figname, save_figure_directory, save_figure_exensions)

        # Show
        if show:
            plt.show()

    # ----------------------------------------
    def mimic_experiments(self, LB, UB, 
                          nbatches:int=5, 
                          noise_mode:str='percentage', 
                          noise_percentage:float=2.5):
        '''Run the experiments with the identified parameters. No inputs required. Stores some attributes:

        Args:
            LB (array, required): Lower bounds of the initial conditions.
            UB (array, required): Upper bounds of the initial conditions.
            nbatches (int, optional): Number of samples to generate. Defaults to 5.
            noise_mode (str, optional): Mode of noise addition. Defaults to 'percentage'.
            noise_percentage (float, optional): Percentage of noise to add. Defaults to 2.5.

        Stores: 
            :y (dict): Dictionary with ground truth data for each batch.
            :y_noisy (dict): Dictionary with noisy data for each batch.
        '''

        # Store information
        self.LB = LB
        self.UB = UB
        self.nbatches = nbatches
        self.noise_mode = noise_mode
        self.noise_percentage = noise_percentage

        # Create dictionary with results
        self.y = {}
        self.y_noisy = {}

        # Record time to solve all ODEs
        self.runtime_odes = 0

        # Generate initial conditions
        samples = pyDOE.lhs(len(LB), self.nbatches, criterion = "maximin")
        self.initial_conditions = LB + (UB - LB)*samples

        # Define time vector
        tspan = self.t

        # Solve ODE
        for sample in range(self.nbatches):
            # Set initial conditions in object
            y0 = self.initial_conditions[sample,:]
            # Solve ODE for noise-free data
            ysample = self.solve_ode(fun=self.model, y0=y0, tspan=tspan, params=self.xopt)
            # Generate noisy data
            ysample_noisy = self.addnoise_per_species(ysample)
            # Put noise free data into dataframe
            self.y[sample] = pd.DataFrame(np.hstack((tspan.reshape(-1,1), ysample)), columns=['time']+self.species)
            # Put noisy data into dataframe
            self.y_noisy[sample] = pd.DataFrame(np.hstack((tspan.reshape(-1,1), ysample_noisy)), columns=['time']+self.species)
        
        # Print message to user
        print(f'[+] Mimiced {self.nbatches} experiments based on the identified parameters.')
         
    # ----------------------------------------   
    def addnoise_per_species(self, y):
        '''Uses some ground truth data and adds some noise to it. Stores the resulting vector as a new attribute. 

        Args: 
            y : Ground truth data. Shape as [npoints_per_batch, nspecies].   

        Stores:
            :y_noisy (array): Noisy observations.
        '''
        y_noisy = np.zeros((y.shape[0], 0))
        for spec_id in range(len(self.species)):
            if self.noise_mode == 'percentage':
                rmse = mean_squared_error(y[:, spec_id], np.zeros(y[:, spec_id].shape), squared=False)
                y_noisy_spec = y[:, spec_id] + np.random.normal(0, rmse / 100.0 * self.noise_percentage, y[:, spec_id].shape)
                y_noisy = np.hstack((y_noisy, y_noisy_spec.reshape(-1,1)))
            else:
                raise ValueError('[-] No other method of noise addition defined here.')

        return y_noisy

    # ----------------------------------------
    def plot_experiments(self, 
                         show:bool=True,
                         save:bool=False, 
                         figname:str='figure',
                         save_figure_directory:str='./figures', 
                         save_figure_exensions:list=['svg','png']):
        '''Plot experiments that were simulated.

        Args:
            show (bool, optional): Boolean indicating whether the figure should be shown. Default to True.
            figname (str, optiona): Name of the figure. Default to 'figure'.
            save (bool, optional): Boolean indicating whether the figure should be saved. Default to False.
            save_figure_directory (str, optional): Directory in which the figure should be saved. Default to './figures'.
            save_figure_exensions (list, optional): List of file extensions in which the figure should be saved. Default to ['svg','png'].
        '''

        # Maximum number of subplots per row
        MAX_SUBPLOTS_PER_ROW = 4
        
        # If more than MAX_SUBPLOTS_PER_ROW*2 species, give warning to that code adjustment is needed and expand MAX_SUBPLOTS_PER_ROW
        if len(self.species) > MAX_SUBPLOTS_PER_ROW*2:
            more_subplots_needed = len(self.species) - MAX_SUBPLOTS_PER_ROW*2
            print('[-] The number of species is larger than {:.0f}. Allowing more subplots now (per row), but codes might need adjustment to make nicer plots (i.e., more rows).'.format(MAX_SUBPLOTS_PER_ROW*2))
            MAX_SUBPLOTS_PER_ROW = len(self.species) + more_subplots_needed/2

        # Create figure
        if len(self.species) > MAX_SUBPLOTS_PER_ROW:
            NROWS = 2
            NCOLS = int(np.ceil(len(self.species)/2))
        else:
            NROWS = 1
            NCOLS = len(self.species)
        fig, ax = plt.subplots(ncols=NCOLS, nrows=NROWS, figsize=(10,5))

        # In case there is only one species, ax is not a list but a single axis
        if len(ax)==1:
            ax = [ax]

        # Define name of time vector
        timename = self.name_of_time_vector

        # Plot ground data
        if NROWS == 1:
            for batch in range(self.nbatches):
                for spec_id, spec in enumerate(self.species): 
                    ax[spec_id].plot(self.y[batch][timename], self.y[batch][spec], color='black', marker='', linestyle='--')
                    ax[spec_id].plot(self.y_noisy[batch][timename], self.y_noisy[batch][spec], marker='o', linestyle='')
                    ax[spec_id].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                    ax[spec_id].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')
        else:
            for batch in range(self.nbatches):
                used_subplot_list = []
                for spec_id, spec in enumerate(self.species): 
                    if spec_id <= MAX_SUBPLOTS_PER_ROW:
                        row = 0
                        col = spec_id
                    else:
                        row = 1
                        col = spec_id - int(len(self.species)/2) - 1
                    ax[row,col].plot(self.y[batch][timename], self.y[batch][spec], color='black', marker='', linestyle='--')
                    ax[row,col].plot(self.y_noisy[batch][timename], self.y_noisy[batch][spec], marker='o', linestyle='')
                    ax[row,col].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                    ax[row,col].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')
                    # Record used subplots
                    used_subplot_list.append([row,col])

            # 'Turn off' unused subplots
            for row in range(NROWS):
                for col in range(NCOLS):
                    if [row,col] not in used_subplot_list:
                        fig.delaxes(ax[row,col])

        # Layout
        plt.tight_layout()
        
        # Save figure if required
        self.save_figure(save, figname, save_figure_directory, save_figure_exensions)

        # Show
        if show:
            plt.show()
    
    # ----------------------------------------
    def plot_train_test_experiments(self, 
                                    show:bool=True,
                                    save:bool=False, 
                                    figname:str='figure',
                                    save_figure_directory:str='./figures', 
                                    save_figure_exensions:list=['svg','png']):
        '''Plot experiments that were simulated. The training and testing runs are colored differently.

        Args:
            show (bool, optional): Boolean indicating whether the figure should be shown. Default to True.
            figname (str, optiona): Name of the figure. Default to 'figure'.
            save (bool, optional): Boolean indicating whether the figure should be saved. Default to False.
            save_figure_directory (str, optional): Directory in which the figure should be saved. Default to './figures'.
            save_figure_exensions (list, optional): List of file extensions in which the figure should be saved. Default to ['svg','png'].
        '''

        # Check that the train-test-split was executed first
        # If traindata is there, the split was executed
        if 'traindata' not in dir(self): 
            raise ValueError('[-] To export the train data, execute train_test_split() first!')

        # Maximum number of subplots per row
        MAX_SUBPLOTS_PER_ROW = 4
        
        # If more than MAX_SUBPLOTS_PER_ROW*2 species, give warning to that code adjustment is needed and expand MAX_SUBPLOTS_PER_ROW
        if len(self.species) > MAX_SUBPLOTS_PER_ROW*2:
            more_subplots_needed = len(self.species) - MAX_SUBPLOTS_PER_ROW*2
            print('[-] The number of species is larger than {:.0f}. Allowing more subplots now (per row), but codes might need adjustment to make nicer plots (i.e., more rows).'.format(MAX_SUBPLOTS_PER_ROW*2))
            MAX_SUBPLOTS_PER_ROW = len(self.species) + more_subplots_needed/2

        # Create figure
        if len(self.species) > MAX_SUBPLOTS_PER_ROW:
            NROWS = 2
            NCOLS = int(np.ceil(len(self.species)/2))
        else:
            NROWS = 1
            NCOLS = len(self.species)
        fig, ax = plt.subplots(ncols=NCOLS, nrows=NROWS, figsize=(10,5))

        # Define name of time vector
        timename = self.name_of_time_vector

        # Plot training batches
        if NROWS == 1:
            for batch in self.traindata:
                for spec_id, spec in enumerate(self.species): 
                    ax[spec_id].plot(self.traindata[batch][timename], self.traindata[batch][spec], color='black', marker='', linestyle='--')
                    labltrain = 'train' if spec_id == 0 and batch == 0 else '__no_label__'
                    ax[spec_id].plot(self.traindata_noisy[batch][timename], self.traindata_noisy[batch][spec], color='blue', alpha=0.7, marker='o', linestyle='', label=labltrain)
                    ax[spec_id].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                    ax[spec_id].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')
        else:
            for batch in self.traindata:
                used_subplot_list = []
                for spec_id, spec in enumerate(self.species): 
                    if spec_id <= MAX_SUBPLOTS_PER_ROW:
                        row = 0
                        col = spec_id
                    else:
                        row = 1
                        col = spec_id - int(len(self.species)/2) - 1
                    ax[row,col].plot(self.traindata[batch][timename], self.traindata[batch][spec], color='black', marker='', linestyle='--')
                    labltrain = 'train' if spec_id == 0 and batch == 0 else '__no_label__'
                    ax[row,col].plot(self.traindata_noisy[batch][timename], self.traindata_noisy[batch][spec], color='blue', alpha=0.7, marker='o', linestyle='', label=labltrain)
                    ax[row,col].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                    ax[row,col].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')
                    # Record used subplots
                    used_subplot_list.append([row,col])
            

        # Plot testing batches
        if NROWS == 1:
            for batch in self.testdata:
                for spec_id, spec in enumerate(self.species): 
                    ax[spec_id].plot(self.testdata[batch][timename], self.testdata[batch][spec], color='black', marker='', linestyle='--')
                    labltest = 'test' if spec_id == 0 and batch == 0 else '__no_label__'
                    ax[spec_id].plot(self.testdata[batch][timename], self.testdata_noisy[batch][spec], color='red', alpha=0.7, marker='d', linestyle='', label=labltest)
                    ax[spec_id].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                    ax[spec_id].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')
        else:
            for batch in self.testdata:
                used_subplot_list = []
                for spec_id, spec in enumerate(self.species): 
                    if spec_id <= MAX_SUBPLOTS_PER_ROW:
                        row = 0
                        col = spec_id
                    else:
                        row = 1
                        col = spec_id - int(len(self.species)/2) - 1
                    ax[row,col].plot(self.testdata[batch][timename], self.testdata[batch][spec], color='black', marker='', linestyle='--')
                    labltest = 'test' if spec_id == 0 and batch == 0 else '__no_label__'
                    ax[row,col].plot(self.testdata[batch][timename], self.testdata_noisy[batch][spec], color='red', alpha=0.7, marker='d', linestyle='', label=labltest)
                    ax[row,col].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                    ax[row,col].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')
                    # Record used subplots
                    used_subplot_list.append([row,col])
            # 'Turn off' unused subplots
            for row in range(NROWS):
                for col in range(NCOLS):
                    if [row,col] not in used_subplot_list:
                        fig.delaxes(ax[row,col])

        # Add legend
        if NROWS == 1:
            ax[0].legend(frameon=False, loc='best')
        else:
            ax[0,0].legend(frameon=False, loc='best')

        # Layout
        plt.tight_layout()
        
        # Save figure if required
        self.save_figure(save, figname, save_figure_directory, save_figure_exensions)

        # Show
        if show:
            plt.show()

    # ----------------------------------------
    def save_figure(self, 
                    save:bool=False, 
                    figure_name:str='figure', 
                    savedirectory:str='./figures', 
                    save_figure_exensions:list=['svg','png']):
        """Saves a figure.

        Args:
            save (bool): Boolean indicating whether the figure should be saved. Defaults to False
            figure_name (str): Name of the figure. Defaults to 'figure'.
            savedirectory (str): Directory in which the figure should be saved. Defaults to './figures'.
            save_figure_exensions (list): List of file extensions in which the figure \
                should be saved. Defaults to ['svg','png'].
        """
    
        if save:
            print(f'[+] Saving figure:')
            if isinstance(save_figure_exensions, list):
                figure_extension_list = save_figure_exensions
            else:
                raise ValueError('[-] The indicated file extension for figures needs to be a list!')
            for figure_extension in figure_extension_list:
                savepath = os.path.join(savedirectory, figure_name+'.'+figure_extension)
                plt.savefig(savepath)
                print(f'\t->{figure_extension}: {savepath}')
        else:
            print(f'[+] Figures not saved.')
    
    # ----------------------------------------
    def export_dict_data_to_excel(self, destination:str='./data', which_dataset:str='all'):
        '''Exports the datasets stored in the dictionary to an excel file. The filename of the data is 
        '{self.example_name}_{which_dataset}.xlsx'
        and '{self.example_name}_{which_dataset}_noisy.xlsx'. 
        
        Args:
            destination : Destination folder in which the excel files should be saved. Default to '.\data'.
            which_dataset : Which dataset should be exported ('all', 'training', or 'testing'). Defaults to 'all'. 
        '''

        # Data to export
        if which_dataset == 'all':
            data_to_export = [self.y, self.y_noisy]
        elif which_dataset == 'training':
            if 'traindata' not in dir(self): 
                raise ValueError('[-] To export the train data, execute train_test_split() first!')
            if self.traindata == {}:
                print('[!] WARNING: Train data is empty! Adjust train_test_split ratio!')
            data_to_export = [self.traindata, self.traindata_noisy]
        elif which_dataset == 'testing':    
            if 'testdata' not in dir(self): 
                raise ValueError('[-] To export the test data, execute train_test_split() first!')
            if self.testdata == {}:
                print('[!] WARNING: Test data is empty! Adjust train_test_split ratio!')
            data_to_export = [self.testdata, self.testdata_noisy]

        # Define filenames
        filename = f'{destination}\{self.example}_{which_dataset}'
        filename_noisy= filename + '_noisy'
        
        # Iteratre through all batches
        for ds, DS in enumerate(data_to_export):
            for batch in DS:

                # Get batchdata
                ybatch = DS[batch]

                # Save noise free to sheet
                if ds == 0:
                    if batch == 0:
                        with pd.ExcelWriter(filename + '.xlsx') as writer:  
                            ybatch.to_excel(writer, sheet_name=f'{batch}') 
                    else:
                        with pd.ExcelWriter(filename + '.xlsx', mode = 'a', engine='openpyxl', if_sheet_exists = 'overlay') as writer:  
                            ybatch.to_excel(writer, sheet_name=f'{batch}')
                if ds == 1:
                    # Save noisy to sheet
                    if batch == 0:
                        with pd.ExcelWriter(filename_noisy + '.xlsx') as writer:  
                            ybatch.to_excel(writer, sheet_name=f'{batch}') 
                    else:
                        with pd.ExcelWriter(filename_noisy + '.xlsx', mode = 'a', engine='openpyxl', if_sheet_exists = 'overlay') as writer:  
                            ybatch.to_excel(writer, sheet_name=f'{batch}')

        # Print message that it worked
        print('[+] Exported batch data to excel.')
        print('\t-> Dataset: ' + which_dataset.upper() + ' (options: training, testing, all)')
        print('\t-> Noise free data to: ' + filename + '.xlsx')
        print('\t-> Noisy data to: ' + filename_noisy + '.xlsx')

    # ----------------------------------------
    def train_test_split(self, test_splitratio=None): 
        """Splits the data into training and testing data.

        Args:
            test_splitratio (float, optional): Ratio [0,1) of the data which is used as test set. \
                Defaults to None (explicitly ask user due to assert).
        """

        # Check that num_train_batches is not None and not larger than number of batches
        assert test_splitratio is not None, '[-] Please specify the ratio [0, 1) used for testing.'
        assert test_splitratio < 1, '[-] The ratio needs to be lower than 1.'
        assert test_splitratio >= 0, '[-] The ratio needs to be larger or equal to 0.'

        # Translate to how many training batches are  needed
        num_train_batches = int(self.nbatches * (1 - test_splitratio))

        # Warn user in case number of test batches is 0 like this
        if num_train_batches == self.nbatches:
            print('[!] Warning: The number of training batches is equal to number of total batches. Using at least one batch for testing now!')
            num_train_batches = self.nbatches - 1
        elif num_train_batches == 0:
            print('[!] Warning: The number of training batches is 0. All batches are used for testing. Using at least one batch for training now!')
            num_train_batches = 1

        # Choose randomly N number of batches for testing
        batches_list = list(self.y.keys())
        self.batchnumbers_test = random.choices(batches_list, k=self.nbatches - num_train_batches)
        self.batchnumbers_test = sorted(self.batchnumbers_test)
        self.batchnumbers_train = [b for b in batches_list if b not in self.batchnumbers_test] 

        # Create empty dictionaries
        self.rawdata = self.y
        self.rawdata_noisy = self.y_noisy
        self.traindata = {}
        self.testdata = {}
        self.traindata_noisy = {}
        self.testdata_noisy = {}

        # Assign data accordingly
        trainindex, testindex = 0, 0
        for batch in batches_list:
            if batch in self.batchnumbers_train:
                self.traindata[trainindex] = self.y[batch]
                self.traindata_noisy[trainindex] = self.y_noisy[batch]
                trainindex += 1
            else:
                self.testdata[testindex] = self.y[batch]
                self.testdata_noisy[testindex] = self.y_noisy[batch]
                testindex += 1





