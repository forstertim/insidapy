'''
Within the simulator, one class is defined as the 'fedbatch' (relies on the 'batch' class)
It is used to simulate fedbatch reactors with defined kinetics.

To see the available default examples that are implemented, load the class and run the show_implemented_examples() method.
'''
from insidapy.ode import fedbatch

#%% ODE examples
data = fedbatch(example='fedbatch1',                                        # Choose example. Defaults to "fedbatch1".
                nbatches=4,                                                 # Number of batches. Defaults to 3.
                npoints_per_batch=20,                                       # Number of points per batch and per species. Defaults to 20.
                noise_mode='percentage',                                    # Noise mode. Defaults to "percentage".
                noise_percentage=2.5,                                       # Noise percentage (in case mode is "percentage")      
                random_seed=10,                                             # Random seed for reproducibility. Defaults to 0.
                # bounds_initial_conditions=[[0.1, 50, 0], [0.4, 90, 0]],   # Bounds for initial conditions. Defaults to "None".
                time_span=[0, 80],                                          # Time span for integration. Defaults to "None". 
                initial_condition_generation_method='LHS',                  # Method for generating initial conditions. Defaults to "LHS".
                name_of_time_vector='time')                                 # Name of time vector. Defaults to "time".

# print available default examples
data.show_implemented_examples()                                            # Print available examples

# Print info
data.print_info()                                                           # Print info

# Run experiments
data.run_experiments()                                                      # Run experiments

# Split into training and testing
data.train_test_split(test_splitratio=0.2)                                  # split into train and test set 

# Plot experiments
data.plot_experiments(save=False, 
                      show= True, 
                      figname=f'{data.example}_simulated_fedbatches',
                      save_figure_directory=r'.\figures', 
                      save_figure_exensions=['png'])                        # plot all batches and save figure

# Plot experiments and make training and testing runs visually distinguishable
data.plot_train_test_experiments(   save=False, 
                                    show=True,
                                    figname=f'{data.example}_simulated_fedbatches_train_test',
                                    save_figure_directory=r'.\figures', 
                                    save_figure_exensions=['png'])          # visualize training and testing batches differently

# Export data
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='all')          # export all data to excel
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='training')     # export training data to excel
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='testing')      # export testing data to excel