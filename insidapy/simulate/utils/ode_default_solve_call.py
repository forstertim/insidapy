
import numpy as np
from scipy.integrate import solve_ivp

from .ode_collection import *     # Imports all the ODEs from the collection file.

def solve_ode(self, y0, tspan):
    """Default funtion calls for solving the appropriate ODE system.
    
    Args:
        y0 (array, required): The initial values for solving the system of ODEs.
        tspan (list, required): The starting and end time point.
        
    Returns:
        array: y - Concentration profiles of the species of shape \
            [n,s] (n being the number of samples and s being the number of species).

    """

    # ..........................................
    if self.example == 'batch1':
        y = solve_ivp(fun=batch1, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
    # ..........................................
    elif self.example == 'batch2':
        y = solve_ivp(fun=batch2, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
    # ..........................................
    elif self.example == 'batch3':
        y = solve_ivp(fun=batch3, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
    # ..........................................
    elif self.example == 'batch4':
        y = solve_ivp(fun=batch4, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
    # ..........................................
    elif self.example == 'batch5':
        y = solve_ivp(fun=batch5, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
    # ..........................................
    elif self.example == 'batch6':
        y = solve_ivp(fun=batch6, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
        
    # ..........................................
    elif self.example == 'fedbatch1':
        y = solve_ivp(fun=fedbatch1, y0=y0, t_span=[tspan[0], tspan[-1]], t_eval=tspan)
        
    # ..........................................
    # Get only species data
    # ..........................................
    y = y.y.T
    return y