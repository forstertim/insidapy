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
        nruns : Number of runs to generate different functions. 
        npoints : Number of x-data points (abscissa). Defalt to 20.
        tspan : List of staring and end points of abscissa data. Default to [0,10].
        noise_mode : Type of noise that should be added. Default to 'percentage'.
        noise_percentage : In case 'percentage' is chosen as noise_mode, this indicates the percentage of \
            noise that is added to the ground truth. Default to 5%.
        seed : The numpy random seed is not fixed by default, but it can be if the user gives an integer value.

    Stores:
        :x (array): Ground truth abscissa data.
        :y (dict): Ground truth function data y(x).
        :y_noisy (dict): Noisy observations of y(x).
    '''
    
    # ----------------------------------------
    def __init__(self, example:str='logistic', 
                 example_args:dict=None,
                 nruns:int=1,
                 npoints:int=20, 
                 tspan:list=[0,10], 
                 noise_mode:str='percentage', 
                 noise_percentage:float=5,
                 seed=None):
        """Initiator function.
        """

        # Store inputs
        self.noise_mode = noise_mode    
        self.tspan = tspan
        self.nruns = nruns
        self.npoints = npoints
        self.noise_percentage = noise_percentage
        self.seed = seed
        
        # Define a seed if the user wants to
        np.random.seed(self.seed)
        if seed is not None:
            print(f'[!] Numpy random seed was fixed by user (seed={self.seed})!')

        # Define abscissa
        self.x = np.linspace(tspan[0], tspan[1], npoints)

        # Define which example function to use
        self.example = example
        self.example_args = example_args
        if example=='sin':
            self.sin()
        elif example=='logistic':
            self.logistic()
        elif example=='step':
            self.step()
        else:
            raise ValueError('[-] No other example defined here.')

        # Add noise
        self.addnoise()

    # ----------------------------------------
    def sin(self):
        '''Sinusoidal ground truth data.

        Stores:
            :y (dict): Ground truth function data y(x).
        '''
        self.y = {}
        for i in range(self.nruns):
            self.y[i] = np.sin(self.x).reshape(-1,1)

    # ----------------------------------------
    def logistic(self):
        '''Logistic function as ground truth data.

        Stores:
            :y (dict): Ground truth function data y(x).
        '''
        self.y = {}
        for i in range(self.nruns):
            self.y[i] = 1/(1 + np.exp(-self.x)).reshape(-1,1)

    # ----------------------------------------
    def step(self):
        '''Step function as ground truth data with the step being located
        at the given position 'step_pos'. Before the step, the function has 
        the value of 'x_min' and after the step the value of 'x_max'.

        Stores:
            :y (dict): Ground truth function data y(x).
        '''
        if self.example_args is None:
            self.example_args = {'step_pos': 1.5, 'x_min': 0.2, 'x_max': 2.1}
            print('[!] The default values were changed. Using the following information for the step function:')
            print('\t->Step location: x={}'.format(self.example_args['step_pos']))
            print('\t->Step from: x_min={}'.format(self.example_args['x_min']))
            print('\t->Step to: x_max={}'.format(self.example_args['x_max']))
        self.y = {}
        for i in range(self.nruns):
            self.y[i] = np.ones(self.x.shape).reshape(-1,1)*self.example_args['x_min']
            self.y[i][self.x > self.example_args['step_pos']] = self.example_args['x_max']

    # ----------------------------------------
    def addnoise(self):
        '''Add noise to the ground truth data.

        Stores:
            :y_noisy (dict): Noisy data of the ground truth data data y(x).
        '''
        # Iterate over nruns
        self.y_noisy = {}
        for i in range(self.nruns):
            if self.noise_mode == 'percentage':
                rmse = mean_squared_error(self.y[i], np.zeros(self.x.shape), squared=False)
                self.y_noisy[i] = self.y[i] + np.random.normal(0, rmse / 100.0 * self.noise_percentage, self.y[i].shape)
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
        for i in range(self.nruns):
            if i == 0: # for this univariate module, currently, the ground truth is always the same
                plt.plot(self.x, self.y[i], color='black', marker='', linestyle='--', label='ground truth')
            plt.plot(self.x, self.y_noisy[i], marker='o', linestyle='', label=f'noisy run {i}')
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

    # ----------------------------------------
    def train_test_split(self, test_splitratio=None, number_list_test=None): 
        """Splits the data into training and testing data.

        Args:
            test_splitratio (float, optional): Ratio [0,1) of the data which is used as test set. \
                Defaults to None (explicitly ask user due to assert).
            number_list_test (list, optional): List of numbers of the runs which are used as test set. \
                If not indicated, the test runs are chosen randomly.
        """

        # Checks
        assert self.nruns > 1, '[-] The number of runs needs to be larger than 1 to do a train-test split.'
        assert test_splitratio is not None, '[-] Please specify the ratio [0, 1) used for testing.'
        assert test_splitratio < 1, '[-] The ratio needs to be lower than 1.'
        assert test_splitratio >= 0, '[-] The ratio needs to be larger or equal to 0.'

        # Translate to how many training runs are needed
        num_train_runs = int(self.nruns * (1 - test_splitratio))

        # Warn user in case number of test runs is 0 like this
        if num_train_runs == self.nruns:
            print('[!] Warning: The number of training runs is equal to number of total runs. Using at least one run for testing now!')
            num_train_runs = self.nruns - 1
        elif num_train_runs == 0:
            print('[!] Warning: The number of training runs is 0. All runs are used for testing. Using at least one run for training now!')
            num_train_runs = 1

        # Choose randomly N number of runs for testing
        runs_list = list(self.y.keys())
        if number_list_test is None:
            self.runnumbers_test = list(np.random.choice(runs_list, size=self.nruns - num_train_runs, replace=False))
        else:
            self.runnumbers_test = number_list_test
        self.runnumbers_test = sorted(self.runnumbers_test)
        self.runnumbers_train = [b for b in runs_list if b not in self.runnumbers_test] 

        # Create empty dictionaries
        self.rawdata = self.y
        self.rawdata_noisy = self.y_noisy
        self.traindata = {}
        self.testdata = {}
        self.traindata_noisy = {}
        self.testdata_noisy = {}

        # Assign data accordingly
        trainindex, testindex = 0, 0
        for run in runs_list:
            if run in self.runnumbers_train:
                self.traindata[trainindex] = self.y[run]
                self.traindata_noisy[trainindex] = self.y_noisy[run]
                trainindex += 1
            else:
                self.testdata[testindex] = self.y[run]
                self.testdata_noisy[testindex] = self.y_noisy[run]
                testindex += 1
            
    # ----------------------------------------
    def export_to_excel(self, destination:str='./data', which_dataset:str='all'):
        '''Exports the datasets stored in the dictionary to an excel file. The filename of the data is 
        'univariate_{self.example}_{which_dataset}.xlsx' and ''univariate_{self.example}_{which_dataset}_noisy.xlsx'. 
        
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
        filename = f'{destination}/univariate_{self.example}_{which_dataset}'
        filename_noisy= filename + '_noisy'
        
        # Iteratre through all batches
        for ds, DS in enumerate(data_to_export):
            for batch in DS:

                # Get batchdata
                colname = ['y' if ds == 0 else 'y_noisy']
                ybatch = pd.DataFrame(np.hstack((self.x.reshape(-1,1), DS[batch])), columns = ['x']+colname)

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