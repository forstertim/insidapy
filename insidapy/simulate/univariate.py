import os
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
import pandas as pd
import pyDOE as lh
import matplotlib.pyplot as plt
import random

class univariate_examples():
    '''Class to initialize example data consisting of a ground truth (noiseless) and noisy observations.
    
    Args:
        example : String of the example to be loaded. Default to 'logistic'.
        npoints : Number of x-data points (abscissa). Defalt to 20.
        tspan : List of staring and end points of abscissa data. Default to [0,10].
        noise_mode : Type of noise that should be added. Default to 'percentage'.
        noise_percentage : In case 'percentage' is chosen as noise_mode, this indicates the percentage of \
            noise that is added to the ground truth. Default to 5%ç.

    Stores:
        :x (array): Ground truth abscissa data.
        :y (array): Ground truth function data y(x).
        :y_noisy (array): Noisy observations of y(x).
    '''
    
    # ----------------------------------------
    def __init__(self, example:str='logistic', 
                 npoints:int=20, 
                 tspan:list=[0,10], 
                 noise_mode:str='percentage', 
                 noise_percentage:float=5):
        """Initiator function.
        """

        # Store inputs
        self.noise_mode = noise_mode    
        self.tspan = tspan
        self.npoints = npoints
        self.noise_percentage = noise_percentage

        # Define abscissa
        self.x = np.linspace(tspan[0], tspan[1], npoints)

        # Define which example function to use
        self.example = example
        if example=='sin':
            self.sin()
        elif example=='logistic':
            self.logistic()
        else:
            raise ValueError('[-] No other example defined here.')

        # Add noise
        self.addnoise()

    # ----------------------------------------
    def sin(self):
        '''Sinusoidal ground truth data.

        Stores:
            :y (array): Ground truth function data y(x).
        '''
        self.y = np.sin(self.x)

    # ----------------------------------------
    def logistic(self):
        '''Logistic function as ground truth data.

        Stores:
            :y (array): Ground truth function data y(x).
        '''
        self.y = 1/(1 + np.exp(-self.x))

    # ----------------------------------------
    def addnoise(self):
        '''Add noise to the ground truth data.

        Stores:
            :y_noisy (array): Noisy data of the ground truth data data y(x).
        '''
        if self.noise_mode == 'percentage':
            rmse = mean_squared_error(self.y, np.zeros(self.x.shape), squared=False)
            self.y_noisy = self.y + np.random.normal(0, rmse / 100.0 * self.noise_percentage, self.y.shape)
        else:
            raise ValueError('[-] No other method of noise addition defined here.')
        
    # ----------------------------------------
    def plot(   self,
                show:bool=True,
                save:bool=False, 
                figname:str='figure',
                save_figure_directory:str='./figures', 
                save_figure_exensions:list=['svg','png']):
        '''Plot the example data.

        Args:
            show (bool, optional): Boolean indicating whether the figure should be shown. Default to True.
            figname (str, optiona): Name of the figure. Default to 'figure'.
            save (bool, optional): Boolean indicating whether the figure should be saved. Default to False.
            save_figure_directory (str, optional): Directory in which the figure should be saved. Default to './figures'.
            save_figure_exensions (list, optional): List of file extensions in which the figure should be saved. Default to ['svg','png'].
        '''
        plt.figure(figsize=(10,5))
        plt.plot(self.x, self.y, color='black', marker='', linestyle='--', label='ground truth')
        plt.plot(self.x, self.y_noisy, color='black', marker='o', linestyle='', label='noisy data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best')

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