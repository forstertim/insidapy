import sys
import os
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
import pandas as pd
import pyDOE as lh
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable

from .utils.ode_default_settings import load_default_settings                       # file with all the default settings of the example problems
from .utils.ode_default_solve_call import solve_ode                                 # file with all the default corresponding ODE solvers
from .utils.ode_default_library_description import ode_default_library_description  # file with a list of all the implemented examples

##########################################
##########################################
### Batch simulator
##########################################
##########################################
class batch():
    """
    The batch class is used to simulate reactor in batch operation mode. \
        The examples that are implemented are listed in utils.ode_default_library_description

    Args:
        example (str, optional): String of the example to be loaded. Default to 'fermentation1' (batch fermentation with 3 species).
        nbatches (int, optional): 
        npoints_per_batch (int, optional): Number of x-data points (abscissa). Defalt to 20.
        noise_mode (str, optional): Type of noise that should be added. Default to 'percentage'.
        noise_percentage (float, optional): In case 'percentage' is chosen as noise_mode, this indicates the percentage of \
            noise that is added to the ground truth. Default to 5%.
        random_seed (int, optional): Fix the seed. Default to 0.
        bounds_initial_conditions (_type_, optional): The examples have different bounds for the initial conditions. \
            The defaults are given in the function load_initial_conditions.If the user wants to overwrite \
            these, this can be done here by providing a list of lower and upper bounds. \
            (i.e., [[lb1, lb2, ...], [ub1, ub2, ...]]).
        time_span (_type_, optional): The examples have different time spans. The defaults are given in \
            the function load_initial_conditions. If the user wants to overwrite these, this can be done \
            here by providing a list of starting and  end time points (i.e., [t0, tf]).
        initial_condition_generation_method (str, optional): Method to generate initial conditions. Default to 'LHS'.
        name_of_time_vector (str, optional): Name of the time vector in the dataframe. Default to 'time'.

    Stores:
        :x (array): Ground truth abscissa data.
        :y (function): Ground truth function data y(x).
        :y_noisy (array): Noisy observations of y(x).
    """
    
    # ----------------------------------------
    def __init__(self, example:str='fermentation1', 
                 nbatches:int=3,
                 npoints_per_batch:int=20, 
                 noise_mode:str='percentage', 
                 noise_percentage:float=5,
                 random_seed:int=0,
                 bounds_initial_conditions=None,
                 time_span=None,
                 initial_condition_generation_method:str='LHS',
                 name_of_time_vector:str='time'):
        """Initialize function.
        """

        # Store inputs
        self.example_type = 'batch'
        self.example = example
        self.noise_mode = noise_mode    
        self.npoints_per_batch = npoints_per_batch
        self.nbatches = nbatches
        self.noise_percentage = noise_percentage
        self.initial_condition_generation_method = initial_condition_generation_method
        self.name_of_time_vector = name_of_time_vector
        self.random_seed = random_seed
        self.overwrite_bounds_initial_conditions = bounds_initial_conditions
        self.overwrite_bounds_time_span = time_span

        # Initial conditions and species information
        self.load_initial_conditions()

    # ----------------------------------------
    def show_implemented_examples(self):
        ode_default_library_description(self.example_type)

    # ----------------------------------------
    def run_experiments(self):
        '''Run the experiments. No inputs required. Stores some attributes:

        Stores: 
            :y (dict): Dictionary with ground truth data for each batch.
            :runtime_odes (int): Cumulative time to solve the ODEs of all batches.
        '''

        # Create dictionary with results
        self.y = {}
        self.y_noisy = {}

        # Record time to solve all ODEs
        self.runtime_odes = 0

        # Define time vector
        tspan = np.linspace(self.tspan[0], self.tspan[1], self.npoints_per_batch); 

        # Solve ODE
        for batch in range(self.nbatches):
            # Solve ODE for noise-free data
            ybatch, runtime = self.solve_ODE_model(self.y0[batch], tspan)
            # Generate noisy data
            ybatch_noisy = self.addnoise_per_species(ybatch)
            # Put noise free data into dataframe
            self.y[batch] = pd.DataFrame(np.hstack((tspan.reshape(-1,1), ybatch)), columns=['time']+self.species)
            # Put noisy data into dataframe
            self.y_noisy[batch] = pd.DataFrame(np.hstack((tspan.reshape(-1,1), ybatch_noisy)), columns=['time']+self.species)
            # Record time
            self.runtime_odes += runtime    
        
        # Print message to user
        print(f'[+] Experiments done.')
         
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
    def load_initial_conditions(self):
        '''Load initial conditions for the ODEs.

        Stores:
            :y0 (array): Initial conditions.
        '''
        # In case the user overwrites the bounds, check that the input is correct
        if self.overwrite_bounds_initial_conditions is not None:
            assert isinstance(self.overwrite_bounds_initial_conditions, list) and len(self.overwrite_bounds_initial_conditions) == 2, '[-] The bounds for the species need to be a list of two elements [[lb1, lb2, ...], [ub1, ub2, ...]].'
        if self.overwrite_bounds_time_span is not None:
            assert isinstance(self.overwrite_bounds_time_span, list) and len(self.overwrite_bounds_time_span) == 2, '[-] The time span needs to be a list of two elements [t0, tf].'
            assert self.overwrite_bounds_time_span[0]<self.overwrite_bounds_time_span[1], '[-] The elements of the time span need need to be in increasing order: t0 < tf'

        # Fix the seed
        np.random.seed(self.random_seed)
        
        # Set a flag which will later raise an error in 
        # case an example ID is given which does not exist
        no_example = True
            
        # ..........................................
        # Default values for the examples in the modules
        self, no_example = load_default_settings(self, no_example)
       
        # ..........................................
        if self.example == 'custom':
            # User-defined ODE system in separate function file
            no_example = False
            self.example_info = 'custom made system'
            self.example_reference = 'you <3'
            self.time_unit = self.name_of_time_unit
            self.species_units = self.name_of_species_units
            self.LB = np.array(self.overwrite_bounds_initial_conditions[0])
            self.UB = np.array(self.overwrite_bounds_initial_conditions[1])
            self.tspan = self.overwrite_bounds_time_span
            text_to_print_time, text_to_print_bounds = 'user-defined', 'user-defined'
            
        # ..........................................
        if no_example:
            raise ValueError('[-] No other example defined here.')
        
        # If no user-inputs about bounds and time span are given, use the default settings
        if self.overwrite_bounds_initial_conditions is not None and self.example != 'custom':
            self.LB = np.array(self.overwrite_bounds_initial_conditions[0])
            self.UB = np.array(self.overwrite_bounds_initial_conditions[1])
            text_to_print_bounds = 'user-defined'
        if self.overwrite_bounds_time_span is not None and self.example != 'custom':
            self.tspan = self.overwrite_bounds_time_span
            text_to_print_time = 'user-defined'
            
        # Check that number of upper and lower bounds are equal to the number of species
        # and that the order is [[LB],[UB]] and not [[UB],[LB]]
        assert len(self.LB) == len(self.species), '[-] Incorrect number of lower bounds!'
        assert len(self.UB) == len(self.species), '[-] Incorrect number of upper bounds!'
        for spec_id, spec in enumerate(self.species):
            assert self.LB[spec_id] <= self.UB[spec_id], f'[-] Lower bound of species >{spec}< seems to be larger than the indicated upper bound!'
        
        # Generate samples by LHS
        if self.overwrite_bounds_initial_conditions is not None:
            print(f'[!] IMPORTANT: It seems that you changed the default bounds of the species. Make sure the order of the indicated bounds is the following: {str(self.species)}')
        if self.nbatches == 1:
            print('[!] Warning: You are generating only one batch. Taking the middle point of upper and lower bounds.')
            samples = np.array([self.LB + (self.UB-self.LB)/2]).reshape(1,-1)
        else:
            if self.initial_condition_generation_method == 'LHS':
                samples = lh.lhs(len(self.LB), self.nbatches, criterion = "maximin")
            else:
                raise ValueError('[-] No other method defined here to generate the initial conditions.')

        # Store final initial conditions
        self.y0 = self.LB + (self.UB - self.LB)*samples

    # ----------------------------------------
    def print_info(self):
        """Prints the information about the example that was loaded.
        """
        print(f'[+] Loaded the example {self.example.upper()} with the following properties:')
        # Create table with field names and align them accordingly
        x = PrettyTable()
        x.field_names = ["Property", "Description"]
        x.align['Property'] = "l"
        x.align['Description'] = "l"
        x.min_width = 30
        x.max_width = 70
        # Add information rows
        x.add_row(["Example string", self.example])
        x.add_row(["Example description", self.example_info])
        x.add_row(["Short reference information", self.example_reference])
        x.add_row(["Number of species", len(self.species)])
        x.add_row(["Species names", [*self.species]])
        x.add_row(["Species units", [*self.species_units]])
        x.add_row(["Number of batches", self.nbatches])
        x.add_row(["Number of samples", self.npoints_per_batch])
        x.add_row(["Time span", [*self.tspan]])
        x.add_row(["Time unit", self.time_unit])
        x.add_row(["Noise mode", self.noise_mode])
        if self.noise_mode == 'percentage':
            x.add_row(["Noise percentage", f"{self.noise_percentage}%"])
        x.add_row(["Lower bounds for experiments", [*self.LB,]])
        x.add_row(["Upper bounds for experiments", [*self.UB,]])
        # Display table
        print(x)

    # ----------------------------------------
    def solve_ODE_model(self, y0, tspan):
        '''Solve the ODE model for the given sample and time span. \
            Returns the derivatives (dydt) as an array of shape [n,] and the runtime of the ODE solver.

        Args:
            y0 : Initial conditions.
            tspan : Time span.
        '''

        # Simulate the reactor operation until the selected time tf
        # ! THE ODEs ARE IMPORTED FROM THE SEPARATE ode_collection-FILE!
        starttime = time.time()
        
        # ..........................................
        if self.example == 'custom':
            # Append relative path to filename
            sys.path.append(self.relative_path_custom_ode)
            # Import module from file provided by user
            if self.custom_ode_function_name is not None:
                filename = self.filename_custom_ode
                funcname = self.custom_ode_function_name
                odefunc = getattr(__import__(filename), funcname)
            else:
                raise ValueError('[-] Please provide a function name for the custom ODE system. \
                                 The automatic identification of the function name is under development.')
            # In case we have user-defined arguments for the ODE file
            if self.ode_arguments is not None:
                y = odeint(func=odefunc, y0=y0, t=tspan, args=(self.ode_arguments,))
            else:
                y = odeint(func=odefunc, y0=y0, t=tspan)
        else:
            y = solve_ode(self, y0, tspan)
                
        # Get runtime for the batch
        runtime_batch = time.time() - starttime
        
        # Raise error in case y is not calculated due to wrong example ID
        if 'y' not in locals():
            raise ValueError('[-] No other example defined here.')
        
        return y, runtime_batch

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

        # Create figure
        fig, ax = plt.subplots(ncols=len(self.species),nrows=1, figsize=(10,5))

        # In case there is only one species, ax is not a list but a single axis
        if len(ax)==1:
            ax = [ax]

        # Define name of time vector
        timename = self.name_of_time_vector

        # Plot ground truth
        for batch in range(self.nbatches):
            for spec_id, spec in enumerate(self.species): 
                ax[spec_id].plot(self.y[batch][timename], self.y[batch][spec], color='black', marker='', linestyle='--')
                ax[spec_id].plot(self.y_noisy[batch][timename], self.y_noisy[batch][spec], marker='o', linestyle='')
                ax[spec_id].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                ax[spec_id].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')

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
        
        # Create figure
        fig, ax = plt.subplots(ncols=len(self.species),nrows=1, figsize=(10,5))

        # Define name of time vector
        timename = self.name_of_time_vector

        # Plot training batches
        for batch in self.traindata:
            for spec_id, spec in enumerate(self.species): 
                ax[spec_id].plot(self.traindata[batch][timename], self.traindata[batch][spec], color='black', marker='', linestyle='--')
                labltrain = 'train' if spec_id == len(self.species)-1 and batch == list(self.traindata.keys())[-1] else '__no_label__'
                ax[spec_id].plot(self.traindata_noisy[batch][timename], self.traindata_noisy[batch][spec], color='blue', alpha=0.7, marker='o', linestyle='', label=labltrain)
                ax[spec_id].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                ax[spec_id].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')

        # Plot testing batches
        for batch in self.testdata:
            for spec_id, spec in enumerate(self.species): 
                ax[spec_id].plot(self.testdata[batch][timename], self.testdata[batch][spec], color='black', marker='', linestyle='--')
                labltest = 'test' if spec_id == len(self.species)-1 and batch == list(self.testdata.keys())[-1] else '__no_label__'
                ax[spec_id].plot(self.testdata[batch][timename], self.testdata_noisy[batch][spec], color='red', alpha=0.7, marker='d', linestyle='', label=labltest)
                ax[spec_id].set_ylabel(f'{spec} / {self.species_units[spec_id]}')
                ax[spec_id].set_xlabel(f'{self.name_of_time_vector} / {self.time_unit}')

        ax[-1].legend(frameon=False, loc='best')

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
        '{self.example_type}_{self.example}_{which_dataset}_batchdata.xlsx'
        and '{self.example_type}_{self.example}_{which_dataset}_batchdata_noisy.xlsx'. 
        
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
        filename = f'{destination}\{self.example_type}_{self.example}_{which_dataset}_batchdata'
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



##########################################
##########################################
### Fed-batch simulator
##########################################
##########################################
class fedbatch(batch):
    """
    The fedbatch class is used to simulate reactor in fedbatch operation mode. \
    The class uses the batch class as basis.
    The examples that are implemented are listed in utils.ode_default_library_description.

    Args:
        example (str, optional): String of the example to be loaded. Default to 'fedbatch1' (fed batch fermentation with 3 species).
        nbatches (int, optional): 
        npoints_per_batch (int, optional): Number of x-data points (abscissa). Defalt to 20.
        noise_mode (str, optional): Type of noise that should be added. Default to 'percentage'.
        noise_percentage (float, optional): In case 'percentage' is chosen as noise_mode, this indicates the percentage of \
            noise that is added to the ground truth. Default to 5%.
        random_seed (int, optional): Fix the seed. Default to 0.
        bounds_initial_conditions (_type_, optional): The examples have different bounds for the initial conditions. \
            The defaults are given in the function load_initial_conditions.If the user wants to overwrite \
            these, this can be done here by providing a list of lower and upper bounds. \
            (i.e., [[lb1, lb2, ...], [ub1, ub2, ...]]).
        time_span (_type_, optional): The examples have different time spans. The defaults are given in \
            the function load_initial_conditions. If the user wants to overwrite these, this can be done \
            here by providing a list of starting and  end time points (i.e., [t0, tf]).
        initial_condition_generation_method (str, optional): Method to generate initial conditions. Default to 'LHS'.
        name_of_time_vector (str, optional): Name of the time vector in the dataframe. Default to 'time'.

    Stores:
        :x (array): Ground truth abscissa data.
        :y (function): Ground truth function data y(x).
        :y_noisy (array): Noisy observations of y(x).
    """
    
    # ----------------------------------------
    def __init__(self, example:str='fedbatch1', 
                 nbatches:int=3,
                 npoints_per_batch:int=20, 
                 noise_mode:str='percentage', 
                 noise_percentage:float=5,
                 random_seed:int=0,
                 bounds_initial_conditions=None,
                 time_span=None,
                 initial_condition_generation_method:str='LHS',
                 name_of_time_vector:str='time'):
        """Initialize function.
        """

        # Since it is also a batch reaction, load the init function of the parent class
        super().__init__(   example=example, 
                            nbatches=nbatches,
                            npoints_per_batch=npoints_per_batch,
                            noise_mode=noise_mode,
                            noise_percentage=noise_percentage,
                            random_seed=random_seed,
                            bounds_initial_conditions=bounds_initial_conditions,
                            time_span=time_span,
                            initial_condition_generation_method=initial_condition_generation_method,
                            name_of_time_vector=name_of_time_vector)

        # Ensures that in the 'show_implemented_examples' function
        # only the fedbatch examples are displayed
        # -> The batch-entry of the super-class is overwritten!
        self.example_type = 'fedbatch'



##########################################
##########################################
### Custom ODE simulator
##########################################
##########################################
class custom_ode(batch):
    """The custom_ode class is used to simulate the time evolution of a system defined in a separate ODE file.
    The intention is to model a batch system.
    The user should create a file with the ODEs and provide the path to this file. 
    The file should contain a function which takes the state vector y and time t as inputs and returns the derivatives dydt.

    Args:
        filename_custom_ode (str, required): The file name to the file containing the system of ODEs.
        relative_path_custom_ode (str, required): The relative path to the file containing the system of ODEs.
        custom_ode_function_name (str, required): The name of the function in the file containing the system of ODEs. \
            IF no name is given, the package tries to identify functions with name "ode" in them.
        num_species (int, required): Number of species in the system.
        bounds_initial_conditions (_type_, optional): The examples have different bounds for the initial conditions. \
            The defaults are given in the function load_initial_conditions.If the user wants to overwrite \
            these, this can be done here by providing a list of lower and upper bounds. \
            (i.e., [[lb1, lb2, ...], [ub1, ub2, ...]]).
        time_span (_type_, optional): The examples have different time spans. The defaults are given in \
            the function load_initial_conditions. If the user wants to overwrite these, this can be done \
            here by providing a list of starting and  end time points (i.e., [t0, tf]).
        ode_arguments (dict, optional): Dictionary containing the arguments that are passed to the ODE \
            function with the form >>def ODE(y,t, args)<<. If none are indicated, it is assumed \
            the ODE file has the form >>def ODE(y,t)<<. Defaults to None. 
        name_of_time_unit (str, optional): Name of the time unit. Default to 'h' (hours).
        name_of_species_units (str, optional): Name of the species units. Default to 'g/L' for every species in the system.
        nbatches (int, optional): Number of different initial conditions. Defaults to 1.
        npoints_per_batch (int, optional): Number of x-data points (abscissa). Defalt to 50.
        noise_mode (str, optional): Type of noise that should be added. Default to 'percentage'.
        noise_percentage (float, optional): In case 'percentage' is chosen as noise_mode, this indicates the percentage of \
            noise that is added to the ground truth. Default to 5%.
        random_seed (int, optional): Fix the seed. Default to 0.
        initial_condition_generation_method (str, optional): Method to generate initial conditions. Default to 'LHS'.
        name_of_time_vector (str, optional): Name of the time vector in the dataframe. Default to 'time'.

    Stores:
        :x (array): Ground truth abscissa data.
        :y (function): Ground truth function data y(x).
        :y_noisy (array): Noisy observations of y(x).
    """

    # ----------------------------------------
    def __init__(self, 
                 filename_custom_ode:str, 
                 relative_path_custom_ode:str,
                 species:list,
                 custom_ode_function_name:str=None,
                 bounds_initial_conditions=None,
                 time_span=None,
                 ode_arguments:dict=None,
                 name_of_time_unit:str='h',
                 name_of_species_units:str=None,
                 nbatches:int=1,
                 npoints_per_batch:int=50, 
                 noise_mode:str='percentage', 
                 noise_percentage:float=5,
                 random_seed:int=0,
                 initial_condition_generation_method:str='LHS',
                 name_of_time_vector:str='time'):
        """Initialize function.
        """

        # Since it is also a batch reaction, load the init function of the parent class
        super().__init__(   example='custom', 
                            nbatches=nbatches,
                            npoints_per_batch=npoints_per_batch,
                            noise_mode=noise_mode,
                            noise_percentage=noise_percentage,
                            random_seed=random_seed,
                            bounds_initial_conditions=bounds_initial_conditions,
                            time_span=time_span,
                            initial_condition_generation_method=initial_condition_generation_method,
                            name_of_time_vector=name_of_time_vector)

        # Define path to file where othe ODEs are stored
        self.example_type = 'custom'
        self.filename_custom_ode = filename_custom_ode
        self.relative_path_custom_ode = relative_path_custom_ode
        self.custom_ode_function_name = custom_ode_function_name

        # Get user-defined arguments for the ODE function
        self.ode_arguments = ode_arguments

        # Define user inputs in addition to inputs of batch-class
        self.species = species
        self.name_of_time_unit = name_of_time_unit
        self.name_of_species_units = ['g/L' for _ in range(self.num_species)] if name_of_species_units is None else name_of_species_units
