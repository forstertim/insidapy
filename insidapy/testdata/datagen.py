import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ##########################################
def test_ode_function(t_, y_, p_):
    """Custom ODE system. A batch reactor is modeled with two species. The following system
    is implemented: A <-[k1],[k2]-> B -[k3]-> C

    No arguments are used in this function.

    Args:
        y (array): Concentration of species of shape [n,].
        t (scalar): time.

    Returns:
        array: dydt - Derivative of the species of shape [n,].
    """

    # Variables  
    A = y_[0]
    B = y_[1]
    C = y_[2]

    # Parameters
    k1 = p_[0] #4
    k2 = p_[1] #2
    k3 = p_[2] #6   

    # Rate expressions
    dAdt = k2*B - k1*A
    dBdt = k1*A - k2*B - k3*B
    dCdt = k3*B

    # Vectorization
    dydt = np.array((dAdt, dBdt, dCdt))

    # Return
    return dydt.reshape(-1,)
    
# ##########################################
def generate_test_data(plotting:bool=False, 
                       save:bool=False):

    # Generate example data
    tspan = np.linspace(0, 3, 20)
    y0 = np.array((1, 0, 0))
    rateconstants = np.array([4, 2, 6])
    soln = solve_ivp(test_ode_function, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan, args=(rateconstants,))
    y = soln.y.T
    # Add noise to data
    y_noise = y + np.random.normal(0, 0.04, y.shape)
    # Plot data
    if plotting:
        plt.figure()
        plt.plot(tspan, y, 'k--')
        plt.plot(tspan, y_noise,'o')
        plt.xlabel('Time')
        plt.ylabel('Sampled concentration data')
        plt.legend(['A', 'B', 'C'])
        save_figure(save, figure_name='observed_state_profiles', 
                    save_figure_exensions=['png'])
    return y_noise, tspan, rateconstants

# ##########################################
def save_figure(save:bool=False, 
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