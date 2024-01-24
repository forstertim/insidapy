# Changelog

## Version 0.2.5
* Added a check which examples are implemented to the `__init__` function of the `batch` class.

## Version 0.2.4
* Corrected the way the concentration profiles are returned after using a custom ODE file.
* Created two new example notebooks for the custom ODE case (with and without additional arguments given for the ODE file).

## Version 0.2.3
* Added missing `__init__` file for utility functions in ODE module.
* Changed documentation for RtD.

## Version 0.2.2
* Added `batch6` example.
* Replaced residual `odeint` functions by the `solve_ivp` functions in the built-in `batch` examples.

## Version 0.2.1
* In the `mimic` class, the same plotting, train_test_splitting and exporting functionalities were added as for the `batch` class.
* Replaced all the `odeint` functions by the `solve_ivp` functions.

## Version 0.2.0
* Added the functionality to hand over a custom ODE file and some noisy data. The `fit_and_augment` class in `insidapy.augment.mimic` will the fit the noisy data the model can then be used to generate new data.

## Version 0.1.0
* Made ready for public release.
* Changed the name from "simulator" to "insidapy" (***in***-***si***lico ***da***ta (generation) in ***Py***thon) to prevent possible confusion with other simulators if installed in the same environment.

## Version 0.0.14
* Split the `batch` and `fedbatch` examples in two different classes. 
* Adjusted the wording a bit to distinguish examples in fedbatch or batch operation mode.

## Version 0.0.13
* Included a bioreactor example in fedbatch operation mode.

## Version 0.0.12
* Included an option to show the implemented examples in the ODE module.

## Version 0.0.11
* Added chemical reaction examples.
* Included the option to print the information about the example in a nice table.
* Added extended reference information about the examples, which are also printed to the table.

## Version 0.0.10
* Added michaelis menten kinetics example to ODE module.

## Version 0.0.9
* Added plot-saving options for univariate class.
* Corrected typo in the univariate class.

## Version 0.0.8
* Added another plotting option to visualize training and testing batches easily in the ODE module.

## Version 0.0.7
* Separated the ODEs to have a better overview. Now, a new file called `ode_collection.py` is available.

## Version 0.0.6
* Improved the instructions how to apply a train-test-split in the readme.
* Added a check in the export function of the ODE module that the train-test-split was executed before letting the user export the training or testing data.

## Version 0.0.5
* Included the option to do the entire simulation with custom ODE files that are stored separately by the user. 
* Updated docs accordingly.

## Version 0.0.4
* Included Sphinx documentation and moved the examples folder one level up. Adjusted reame file accordingly.

## Version 0.0.3
* Included a multivariate class with the Rosenbrock function as example. The function data (noisy and noisfree) can be exported as an excel file.

## Version 0.0.2
* Included the option of overwriting the bounds of the initial conditions and the time span for the ODE integration. Renamed the package from "emulator" to "simulator".

## Version 0.0.1
* Creation of the emulator.