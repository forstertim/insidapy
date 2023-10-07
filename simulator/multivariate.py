import os
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
import pandas as pd
import pyDOE as lh
import matplotlib.pyplot as plt
import random

class multivariate_examples():
    '''
    Class to initialize example data consisting of a ground truth (noiseless) and noisy observations.
    
    Args:
        example (str, optional): String of the example to be loaded. Default to 'rosenbrock'.
        npoints (int, optional): Number of grid points per dimension. Defalt to 20.
        noise_mode (str, optional): Type of noise that should be added. Default to 'percentage'.
        noise_percentage (float, optional): In case 'percentage' is chosen as noise_mode, this indicates the percentage of noise that is added to the ground truth. Default to 5%.
        coefs (list, optional): Coefficients of the example function. Default to 'None' (defined in examples below individually).
    '''
    
    # ----------------------------------------
    def __init__(self, 
                 example:str='rosenbrock', 
                 npoints:int=20,
                 noise_mode:str='percentage', 
                 noise_percentage:float=5,
                 coefs=None):
        '''
        Initiator function.
        '''
        
        # Store inputs
        self.noise_mode = noise_mode    
        self.npoints = npoints
        self.noise_percentage = noise_percentage

        # Define which example function to use
        self.example = example
        if example=='rosenbrock':
            self.rosenbrock(coefs)
        else:
            raise ValueError('[-] No other example defined here.')

    # ----------------------------------------
    def rosenbrock(self, coefs=None):        
        """Rosenbrock function. No inputs required. Stores several attributes:

        Stores:
            :x (array): Independent variables as a vector of shape [n,].

            :z (function): Dependent variable y(x) as a lambda function.

            :X, Y (array): Independent variables as a grid of shape [n,n].

            :Z (array): Dependent variable y(X,Y) as a grid of shape [n,n].

            :Z_noiy (array): Noisy observations of y(X,Y) as a grid of shape [n,n].

        """

        
        if coefs is not None: # user coefs
            assert len(coefs)==2, '[-] Coefficients for the Rosenbrock function should be a list of length 2.'
            self.coefs = coefs
        else: # default coefs
            self.coefs = [1, 100]
        # Create grid
        x1 = np.linspace(-2,2,self.npoints)
        x2 = np.linspace(-1,3,self.npoints)
        self.x = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
        self.z = lambda x1, x2: (self.coefs[0] - x1)**2 + self.coefs[1]*(x2 - (x1**2))**2
        # Evaluate function
        self.X, self.Y = np.meshgrid(self.x[:,0], self.x[:,1])
        self.Z = self.z(self.X, self.Y)
        self.independend_variables = [self.X, self.Y]
        # Add noise
        self.addnoise()

    # ----------------------------------------
    def addnoise(self):
        """Add noise to the ground truth data. No inputs required. Stores some attributes:

        Stores:
            :Z_noisy (array): Noisy observations of y(X,Y) as a grid of shape [n,n].
        """
        
        self.Z_noisy = self.Z.copy()
        for var in range(self.Z.shape[1]):
            if self.noise_mode == 'percentage':
                rmse = mean_squared_error(self.Z[:, var], np.zeros(self.Z[:, var].shape), squared=False)
                self.Z_noisy[:,var] = self.Z[:, var] + np.random.normal(0, rmse / 100.0 * self.noise_percentage, self.Z[:, var].shape)
            else:
                raise ValueError('[-] No other method of noise addition defined here.')
            
    # ----------------------------------------
    def contour_plot_2_variables(self, 
                                 nlevels=15,
                                 show:bool=True,
                                 save=False, 
                                 save_figure_directory='./figures', 
                                 save_figure_exensions:list=['svg','png']):
        """Contour plot of the example.

        Args:
            nlevels (int, optional): Number of level curves (integer) or a list of values \
                at which the levels curves should be shown. Default to 15 levels.
            show (bool, optional):Boolean indicating whether the figure should be shown. Default to True.
            save (bool, optional): Boolean indicating whether the figure should be saved. Default to False.
            save_figure_directory (str, optional): Directory in which the figure should be saved. Default to './figures'.
            save_figure_exensions (list, optional): List of file extensions in which the figure should be saved. Default to ['svg','png'].
        """
        
        # Levels
        levels = np.linspace(np.min(self.Z), np.max(self.Z), nlevels)
        levels_noisy = np.linspace(np.min(self.Z_noisy), np.max(self.Z_noisy), nlevels)
        # Plot settings
        alpha_ = 0.7
        figsize_ = (10,4)
        # Plot
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize_)
        cs0 = axes[0].contourf(self.X, self.Y, self.Z, levels=levels, cmap=plt.cm.viridis, alpha=alpha_)
        cs1 = axes[1].contourf(self.X, self.Y, self.Z_noisy, levels=levels_noisy, cmap=plt.cm.viridis, alpha=alpha_)
        # Adjustments
        fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=1, wspace=0.3, hspace=0.02)
        # Labels
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('$y$')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('$y_{noisy}$')
        axes[0].set_title(f'{self.example} (true)')
        axes[1].set_title(f'{self.example} (noisy)')
        # Colorbar
        fig.colorbar(cs1, ax=axes.ravel().tolist(), shrink=0.95, orientation='vertical')
        # Save figure if required
        figname = f'{self.example}'
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
    def export_to_excel(self, destination:str='./data'):
        """Exports the datasets stored in the dictionary to an excel file. The filename is '{example}_data.xlsx'
        and '{example}_data_noisy.xlsx'

        Args:
            destination (str, optional): Destination folder in which the excel files should be saved. Default to '.\data'.
        """

        # Define filenames
        filename = f'{destination}\{self.example}_data'
        filename_noisy= filename + '_noisy'

        # Create dataframes for variables
        df_variables = {}
        for var_ind, var in enumerate(self.independend_variables):
            df_variables[var_ind] = pd.DataFrame(var, columns=[f'x{i}' for i in range(var.shape[1])])

        # Create dataframes for dependent variables
        df_Z = pd.DataFrame(self.Z, columns=[f'z{i}' for i in range(self.Z.shape[1])])
        df_Z_noisy = pd.DataFrame(self.Z_noisy, columns=[f'z{i}' for i in range(self.Z.shape[1])])

        # Create excel file for function data (noisless and noisy)
        with pd.ExcelWriter(filename + '.xlsx') as writer:  
            df_Z.to_excel(writer, sheet_name='Z') 
        
        with pd.ExcelWriter(filename_noisy + '.xlsx') as writer:  
            df_Z_noisy.to_excel(writer, sheet_name='Z_noisy') 

        # Add independent variable data (noisless and noisy)
        for var_ind, var in enumerate(self.independend_variables):
            with pd.ExcelWriter(filename + '.xlsx', mode = 'a', engine='openpyxl', if_sheet_exists = 'overlay') as writer:  
                df_variables[var_ind].to_excel(writer, sheet_name=f'X{var_ind}') 
            
            with pd.ExcelWriter(filename_noisy + '.xlsx', mode = 'a', engine='openpyxl', if_sheet_exists = 'overlay') as writer:  
                df_variables[var_ind].to_excel(writer, sheet_name=f'X{var_ind}') 