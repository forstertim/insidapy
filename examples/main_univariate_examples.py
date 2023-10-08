'''
The univariate_examples function generates univariate data. 
'''

from insidapy.univariate import univariate_examples
        
#%% Univariate examples
univariatedata = univariate_examples(example='sin', 
                                     npoints=20, 
                                     tspan=[0,10], 
                                     noise_mode='percentage', 
                                     noise_percentage=10)
univariatedata.plot(show=True,
                    save=True, 
                    figname='univariate_example',
                    save_figure_directory='./figures', 
                    save_figure_exensions=['png'])
