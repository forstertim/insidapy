In-Silico Data Generation in Python (InSiDa-Py)
================================================
[![Documentation Status](https://readthedocs.org/projects/insidapy/badge/?version=latest)](https://insidapy.readthedocs.io/en/latest/?badge=latest)

This package is used to simplify the generation of example data for different case studies. In several applications, for example in the field of surrogate modeling, it is necessary to generate some *in-silico* data which can be used to train a model. The tool simplifies the generation and export of such data. Some cited applications are included, where the user can easily create custom systems as shown below.

- [Installation](#installation) 💻
- [Included tools](#included-tools) 🧰
- [Examples](#examples) ⚗️
  * [Simulate bioreactor in batch operation mode](#bioreactor-in-batch-operation-mode)
  * [Simulate Rosenbrock function](#rosenbrock-function)
  * [Simulate Custom ODE system](#custom-ode)
  * [Mimic observed batch data](#mimic-batch-data)
- [References](#references) 📚
- [Contribute](#contribute)


Installation
============
If you are a git user, try to install the package using the following command:
```
pip install git+https://github.com/forstertim/insidapy.git
```

Alternatively, you can clone the repository and install it. To easily add your own case studies, install in developer mode (`-e` in the pip install command):
```
git clone https://github.com/forstertim/insidapy.git
cd insidapy
pip install -e .
```

If you do not use git, above, click the button `Code` and select Download ZIP. Unzip the files and store the folder in a place of your choice. Open a terminal inside that folder (you should see the `setup.py` file). Run the following command in the terminal:
```
pip install -e .
```

By using `pip list`, you should see the package installed in your environment (together with the path to the corresponding folder in case you used the developer mode).

Included tools
==============
The simulator includes several options to generate data in `insidapy.simulate`:

* **Univariate**: Data for some univariate functions can be generated. By choosing one of the examples in the univariate class, the method automatically generates the ground truth and noisy data according to the user's input. Currently, the following functions to generate noisy data are implemented:
    * *`sin`* (default): Sinusoidal function $y=\sin(x)$.
    * *`logistic`*: Logistic function $y=1/(1+\exp(x))$.
    * *`step`*: Step function that takes additional inputs (location of the step, y-value before and after the step)

* **Multivariate**: Data for some multivariate functions can be generated. By choosing one of the examples in the multivariate class, the method automatically generates the ground truth and noisy data according to the user's input. Currently, the following functions to generate noisy data are implemented:
    * *`rosenbrock`* (default): The Rosenbrock function is simulated. Useful for optimization benchmarking. The function takes the following form: $f(x,y)=(a-x)^{2}+b(y-x^{2})^{2}$. This function was included in this module after using it in the work of [Forster et al. (2023a)](#references).

* **ODE**: The available classes are `batch`, `fedbatch`, and `custom_ode`. All of them implement an ODE solver for different case studies. Some examples of the implemented case studies are the following (the currently implemted examples are visible by loading the class (`batch` or `fedbatch`) and using the `show_implemented_examples()` function)

    * **`batch`**:
        * *`batch1`* (default): Batch fermentation process with three species based on the work of [Turton et al. (2018)](#references) and used in [Forster et al. (2023b)](#references). 
        * *`batch2`*: Batch fermentation process with four species based on the work of [Del Rio‐Chanona et al. (2019)](#references). 
        * *`batch3`*: Michaelis-Menten kinetics and four species. Example was adapted from [Wong et al. (2023)](#references)
        * *`batch4`*: Chemical reaction or reaction network. Several available, check details with `print_info()`. Most examples were taken from [Floudas et al. (1999)](#references)
        * *`batch5`*: Chemical reaction or reaction network. Several available, check details with `print_info()`.
        * *`batch6`*: Batch fermentation process with seven species based on the work of [Craven et al. (2012)](#references).
    
    * **`fedbatch`**:
        * *`fedbatch1`*: Example was taken from [Seborg et al. (2016)](#references)
   
    * **`custom_ode`**:
        * The user can store a separate function file with a system of ODEs. The function can take additional parameters as inputs or not (i.e., `def ODE(y,t)` or `def ODE(y,t,coefs)`). This allows to easily set up a custom case study. An example is shown below. If not explicitly stated, the module assumes the first structure of the ODE file (no additional inputs).

By choosing one of the examples in this class, the method automatically generates the ground truth and noisy data according to the user's input.

Additionally, the package includes a wrapper for parameter estimation in `insidapy.augment`. The `mimic` class allows to hand over some noisy data and an ODE file. The model parameters are then estimated. The identified model can then be used to generate new data.

Examples
========

## Bioreactor in batch operation mode
The example notebook is stored in `docs/notebooks/main_batch_fermentation_example`. A quick code summary is given here (for more details, check the notebook or the [documentation](https://insidapy.readthedocs.io/en/latest/notebooks/main_batch_examples.html)). 

```python
from insidapy.simulate.ode import batch

data = batch(   example='batch1',
                nbatches=4,
                npoints_per_batch=20,
                noise_mode='percentage', 
                noise_percentage=2.5,
                random_seed=10,
                bounds_initial_conditions=[[0.1, 50, 0], [0.4, 90, 0]],
                time_span=[0, 80],
                initial_condition_generation_method='LHS',
                name_of_time_vector='time')

data.run_experiments()

data.train_test_split(test_splitratio=0.2)

data.plot_experiments(  save=True, 
                        show= False, 
                        figname=f'{data.example}_simulated_batches',
                        save_figure_directory=r'.\figures', 
                        save_figure_exensions=['png'])

data.plot_train_test_experiments(   save=True, 
                                    show=False,
                                    figname=f'{data.example}_simulated_batches_train_test',
                                    save_figure_directory=r'.\figures', 
                                    save_figure_exensions=['png'])
```

![Fig 1. Example several runs in batch operation mode (batch1 example).](https://github.com/forstertim/insidapy/blob/master/docs/notebooks/figures/batch1_simulated_batches.png?raw=true)

*Fig 1. Example several runs in batch operation mode (batch1 example).*

![Fig 2. Example several runs in batch operation mode (batch1 example) with training and testing batches visualized differently.](https://github.com/forstertim/insidapy/blob/master/docs/notebooks/figures/batch1_simulated_batches_train_test.png?raw=true)

*Fig 2. Example several runs in batch operation mode (batch1 example) with training and testing batches visualized differently.*

After the simulation, one can export the data as XLSX files. By choosing `which_dataset` to be `training` (only executable if `train_test_split` was applied), `testing` (only executable if `train_test_split` was applied), or `all` (always executable), the corresponding data is exported to the indicated location:

```python
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='all')      # all data
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='training') # train data (blue circles in Fig 2)
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='testing')  # test data (red diamonds in Fig 2)
```

## Rosenbrock function
The example notebook is stored in `docs/notebooks/main_multivariate_examples.py`. A quick code summary is given here (for more details, check the notebook or the [documentation](https://insidapy.readthedocs.io/en/latest/notebooks/main_multivariate_examples.html)).  

```python
from insidapy.simulate.multivariate import multivariate_examples

data = multivariate_examples(   example='rosenbrock',
                                coefs=[1, 100], 
                                npoints=20, 
                                noise_mode='percentage', 
                                noise_percentage=10)

data.contour_plot_2_variables(  nlevels=15, 
                                show=False,
                                save=True, 
                                save_figure_directory=r'.\figures', 
                                save_figure_exensions=['png'])

data.export_to_excel(destination=r'.\data')
```

![Fig 3. Example of the Rosenbrock function.](https://github.com/forstertim/insidapy/blob/master/docs/notebooks/figures/rosenbrock.png?raw=true)

*Fig 3. Example of the Rosenbrock function.*


## Custom ODE
Two examples are stored in `docs/notebooks`. One of the custom examples includes the additional passing of input arguments to a given ODE file (check out `docs/notebooks/main_custom_ode_with_arguments_passing.ipynb` or the [documentation](https://insidapy.readthedocs.io/en/latest/notebooks/main_custom_ode_with_arguments.html)). The other one shows an example without such additional input arguments for the ODE file (check out `docs/notebooks/main_custom_ode_without_arguments_passing.py` or the [documentation](https://insidapy.readthedocs.io/en/latest/notebooks/main_custom_ode_without_arguments.html)).

A quick summary is given here. The available separate ODE file could look for example like this:
```python
import numpy as np
def customode(t, y, coefs):
        """Custom ODE system. A batch reactor is modeled with two species. The following system
        is implemented: A <-[k1],[k2]-> B -[k3]-> C

        Args:
            y (array): Concentration of species of shape [n,].
            t (scalar): time.
            coefs (dict): Dictionary of coefficients or other information.

        Returns:
            array: dydt - Derivative of the species of shape [n,].
        """
        # Variables  
        A = y[0]
        B = y[1]
        C = y[2]
        # Parameters
        k1 = coefs['k1']
        k2 = coefs['k2']
        k3 = coefs['k3']    
        # Rate expressions
        dAdt = k2*B - k1*A
        dBdt = k1*A - k2*B - k3*B
        dCdt = k3*B
        # Vectorization
        dydt = np.array((dAdt, dBdt, dCdt))
        # Return
        return dydt.reshape(-1,)
```

Similar to the `batch`-class example above, the instance is created and experiments can be run. The data can be plotted and exported:
```python
data = custom_batch_ode(filename_custom_ode=CUSTOM_ODE_FILENAME,                        #Filename of the file containing the ODE system.
                        relative_path_custom_ode=CUSTOM_ODE_RELATIVE_PATH,              #Relative path to the file containing the ODE system.
                        custom_ode_function_name=CUSTOM_ODE_FUNC_NAME,                  #Name of the ODE function in the file.
                        species=CUSTOM_ODE_SPECIES,                                     #List of species.
                        bounds_initial_conditions=CUSTOM_ODE_BOUNDS_INITIAL_CONDITIONS, #Bounds for initial conditions.
                        time_span=CUSTOM_ODE_TSPAN,                                     #Time span for integration.
                        ode_arguments=CUSTOM_ODE_ARGUMENTS,                             #Arguments of the ODE system. Defaults to "None".
                        name_of_time_unit=CUSTOM_ODE_NAME_OF_TIME_UNIT,                 #Name of time unit. Defaults to "h".
                        name_of_species_units=CUSTOM_ODE_NAME_OF_SPECIES_UNITS,         #Name of species unit. Defaults to "g/L".
                        nbatches=3,                                                     #Number of batches. Defaults to 1.
                        npoints_per_batch=50,                                           #Number of points per batch and per species. Defaults to 50.
                        noise_mode='percentage',                                        #Noise mode. Defaults to "percentage".
                        noise_percentage=2.5,                                           #Noise percentage (in case mode is "percentage"). Defaults to 5%.      
                        random_seed=0,                                                  #Random seed for reproducibility. Defaults to 0.
                        initial_condition_generation_method='LHS',                      #Method for generating initial conditions. Defaults to "LHS".
                        name_of_time_vector='time')                                     #Name of time vector. Defaults to "time".

data.run_experiments()
	
data.plot_experiments(  show=True,
                        save=True, 
                        figname='custom_odes_with_args', 
                        save_figure_directory=r'.\figures', 
                        save_figure_exensions=['png'])

data.export_dict_data_to_excel(destination=r'.\data', which_dataset='all')
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='training') 
data.export_dict_data_to_excel(destination=r'.\data', which_dataset='testing')  
```

![Fig 4. Example of the simulation of a custom ODE file.](https://github.com/forstertim/insidapy/blob/master/docs/notebooks/figures/custom_odes_with_args.png?raw=true)

*Fig 4. Example of the simulation of a custom ODE file.*


## Mimic batch data
The example code is stored in `docs/notebooks/main_mimic_observed_batch_data.ipynb` (check out the [documentation](https://insidapy.readthedocs.io/en/latest/notebooks/main_mimic_observed_batch_data.html) for more details). Here, a short summary is given to explain the idea behind mimicing batch data. Assuming some state profiles are observed, we want to use the behaviour to create "look-alike-profiles". Below, we call this process "mimicing the data". Let's say we observed the following profile (dashed lines are the ground-truth behaviour, where the dots are the observed noisy samples):

![Fig 5. Observed noisy state profiles](https://github.com/forstertim/insidapy/blob/master/docs/notebooks/figures/observed_state_profiles.png?raw=true)

*Fig 5. Example of some observed data.*

The identified parameters are used to create new runs that show a similar behaviour as the observed system. Using some bounds for the creation of initial conditions, the following batch data can be generated:

![Fig 6. Generated state profiles using the identified parameters](https://github.com/forstertim/insidapy/blob/master/docs/notebooks/figures/mimiced_experiments_custom_ode.png?raw=true)

*Fig 6. Example of some observed data.*

After running this approach, the same excel-export functionalities as shown in the [bioreactor case study](#bioreactor-in-batch-operation-mode) above can be used.


References
==========
> **Craven S., Shirsat N., Whelan J., Glennon G.**, Process Model Comparison and Transferability Across Bioreactor Scales andModes of Operation for a Mammalian Cell Bioprocess. **2012**. *Biotechnology Progress*. [URL](https://aiche.onlinelibrary.wiley.com/doi/full/10.1002/btpr.1664)

> **Del Rio-Chanona E.A., Cong X., Bradford E., Zhang D., Jing K.**, Review of advanced physical and data-driven models for dynamic bioprocess simulation: Case study of algae–bacteria consortium wastewater treatment. **2019**. *Biotechnology and Bioengineering*. [URL](https://onlinelibrary.wiley.com/doi/abs/10.1002/bit.26881)

> **Floudas C.A., Pardalos P.M., Adjiman C.S., Esposito W.R., Gümüs Z.H**, Handbook of Test Problems in Local and Global Optimization. *In: Series Title: Nonconvex Optimization and Its Applications*. **1999**, Springer US. ISBN 978-1-4419-4812-0.

> **Forster T., Vázquez D., Guillén-Gosálbez G.**, Global optimization of symbolic surrogate process models based on Bayesian learning. **2023a**. *Computer Aided Chemical Engineering*. [URL](https://www.sciencedirect.com/science/article/abs/pii/B9780443152740501980)

> **Forster T., Vázquez D., Cruz-Bournazou M.N., Butté A., Guillén-Gosálbez G.**, Modeling of bioprocesses via MINLP-based symbolic regression of S-system formalisms. **2023b**. *Computers & Chemical Engineering*. [URL](https://www.sciencedirect.com/science/article/pii/S0098135422004410)

> **Seborg D.E., Edgar F.T., Mellichamp D.A., Doyle F.J.**, Process Dynamics and Control, 4th edition. *2016*, Wiley. ISBN: 978-1-119-28591-5.

> **Turton R., Shaeiwtz J.A., Bhattacharyya D., Whiting W.B.**, Analysis, synthesis and design of chemical processes, 5th Edition, **2018**, Prentice Hall. ISBN 0-13-512966-4.

> **Wong S.W.K., Yang S., and Kou S.C.**, Estimating and Assessing Differential Equation Models with Time-Course Data. **2023**. *J Phys Chem B. 2023*. [URL](https://pubs.acs.org/doi/10.1021/acs.jpcb.2c08932)


Contribute
==========
If you have other interesting examples that you would like to have implemented, raise an issue with a reference to the example (i.e., a DOI to a paper with the system). There is a template for such an issue which you can use. Another option is to implement the example yourself and raise a pull request. The same applies to bugs or other issues. 