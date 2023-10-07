
import numpy as np
from scipy.integrate import odeint

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
    if self.example == 'fermentation1':
        y = odeint(func=ODE_Batch_Fermentation_ThreeSpecs, y0=y0, t=tspan)
    # ..........................................
    elif self.example == 'fermentation2':
        y = odeint(func=ODE_Batch_Fermentation_FourSpecs, y0=y0, t=tspan)
    # ..........................................
    elif self.example == 'fedbatch1':
        y = odeint(func=fedbatch1, y0=y0, t=tspan)
    # ..........................................
    elif self.example == 'michaelismenten1':
        y = odeint(func=ODE_Batch_Michaelis_Menten_FourSpecs, y0=y0, t=tspan)
    # ..........................................
    elif self.example == 'chem1':
        y = odeint(func=ODE_Batch_Chemical_Reaction_1, y0=y0, t=tspan)
    # ..........................................
    elif self.example == 'chem2':
        y = odeint(func=ODE_Batch_Chemical_Reaction_2, y0=y0, t=tspan)
        
        
    return y