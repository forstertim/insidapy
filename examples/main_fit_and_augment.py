'''
Here we show an exmaple of estimating parameters in an ODE model and using the fit to generate some similar data profiles. 
'''

# Import some packages
import numpy as np
import matplotlib.pyplot as plt

# Generate some noisy test data
from insidapy.testdata import generate_test_data
y_noise, tspan, rateconstants = generate_test_data(plotting=True)

# If we have an ODE model in which the parameters should be estimated, some bounds for the parameters are
# needed. We therefore set them first. In the example-ODE, we have 3 parameters to be estimated.
# Set the bounds to [0, 5] for each of them.
PARAMBOUNDS = np.zeros((3,2))
PARAMBOUNDS[:,1] = 5
RS_STEPS = 5

# Provide an ODE file with the model
# MAKE SURE THE FUNCTION USES THE FORM f(t,y,params)! Since we use the scipy solver "solve_ivp"!
from customodefile_for_estimation import odemodel

# Instantiate class to fit the noisy data
from insidapy.augment.mimic import fit_and_augment
obj = fit_and_augment(y=y_noise,
                      t=tspan,
                      nparams=len(rateconstants),
                      parameter_bounds=PARAMBOUNDS,
                      model=odemodel)

# Fit the parameters in the ODE model
# -> The fit and prediction can also be done in one step with the fit_predict() function
obj.fit(method='Nelder-Mead', objective='RMSE', num_random_search_steps=RS_STEPS)

# Predict and plot the state profiles (apply the identified estimated parameters)
# -> The fit and prediction can also be done in one step with the fit_predict() function
obj.predict(plotting=True)

# Show the plots 
plt.show()