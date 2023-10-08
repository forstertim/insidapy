'''
Within the simulator, one class of the ODE module is defined as the 'custom_ode'. 
It takes a user-defined ODE system and creates the data to it.
'''

from insidapy.ode import custom_ode

# Define where the ODE file is located and how the ODE function is called
CUSTOM_ODE_RELATIVE_PATH = '.'
CUSTOM_ODE_FILENAME = 'customodefile_without_args'
CUSTIM_ODE_FUNC_NAME = 'customode'

# Give information about the ODE system
CUSTOM_ODE_SPECIES = ['A', 'B', 'C']
CUSTOM_ODE_TSPAN = [0, 3]
CUSTOM_ODE_BOUNDS_INITIAL_CONDITIONS = [[2, 0, 0], [3, 1, 0]]

# Define the units of the ODE system
CUSTOM_ODE_NAME_OF_TIME_UNIT = 'hours'
CUSTOM_ODE_NAME_OF_SPECIES_UNITS = ['g/L', 'g/L', 'g/L']

# Simulate data
data = custom_ode(  filename_custom_ode=CUSTOM_ODE_FILENAME,                        # REQUIRED: Filename of the file containing the ODE system.
                    relative_path_custom_ode=CUSTOM_ODE_RELATIVE_PATH,              # REQUIRED: Relative path to the file containing the ODE system.
                    custom_ode_function_name=CUSTIM_ODE_FUNC_NAME,                  # REQUIRED: Name of the ODE function in the file.
                    species=CUSTOM_ODE_SPECIES,                                     # REQUIRED: List of species.
                    bounds_initial_conditions=CUSTOM_ODE_BOUNDS_INITIAL_CONDITIONS, # REQUIRED: Bounds for initial conditions.
                    time_span=CUSTOM_ODE_TSPAN,                                     # REQUIRED: Time span for integration.
                    name_of_time_unit=CUSTOM_ODE_NAME_OF_TIME_UNIT,                 # OPTIONAL: Name of time unit. Defaults to "h".
                    name_of_species_units=CUSTOM_ODE_NAME_OF_SPECIES_UNITS,         # OPTIONAL: Name of species unit. Defaults to "g/L".
                    nbatches=1,                                                     # OPTIONAL: Number of batches. Defaults to 1.
                    npoints_per_batch=50,                                           # OPTIONAL: Number of points per batch and per species. Defaults to 50.
                    noise_mode='percentage',                                        # OPTIONAL: Noise mode. Defaults to "percentage".
                    noise_percentage=2.5,                                           # OPTIONAL: Noise percentage (in case mode is "percentage"). Defaults to 5%.      
                    random_seed=0,                                                  # OPTIONAL: Random seed for reproducibility. Defaults to 0.
                    initial_condition_generation_method='LHS',                      # OPTIONAL: Method for generating initial conditions. Defaults to "LHS".
                    name_of_time_vector='time')                                     # OPTIONAL: Name of time vector. Defaults to "time".

# Run experiments
data.run_experiments()                                                              # Run experiments

# Plot batches
data.plot_experiments(  show=True,
                        save=True, 
                        figname='custom_odes_without_args', 
                        save_figure_directory=r'.\figures', 
                        save_figure_exensions=['png'])                              # plot all batches and save figure