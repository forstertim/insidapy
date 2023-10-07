from .univariate import univariate_examples
from .multivariate import multivariate_examples
from .ode import batch
from .ode import fedbatch
from .ode import custom_ode

from .utils import ode_collection                   # Imports all the ODEs from the collection file.
from .utils import ode_default_settings             # file with all the default settings of the example problems
from .utils import ode_default_solve_call           # file with all the calls for the default ODEs to be solved
from .utils import ode_default_library_description  # file with all the default IDs with a description