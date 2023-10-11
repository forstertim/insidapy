'''
The multivariate_examples function generates multivariate data. 
'''

from insidapy.simulate.multivariate import multivariate_examples
  
#%% Univariate examples
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