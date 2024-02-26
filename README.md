# MultiMSM

This repository contains a framework for creating a collection of Markov State Models (MSMs) for self-assembling systems with a finite pool of subunits. This is done by leveraging transition data obtained at different values of the fraction of monomers currently in the system.  

The rates of forming various structures depend on the concentration of free monomers, so transition events are augmented with the monomer fraction present when they occur. The user provides some discretization of the interval [0,1] for the monomer fraction, and the library will construct an MSM on each subinterval using transitions that occured for monomer fractions within those bounds. The library also contains optimizers to choose this discretization for you, by best reconstructing the average yields of a target state from your sampling data. 

Using the MSM estimated monomer fraction dynamics, one can solve the forward and backward Kolmogorov equations for this collection of MSMs. We can also aggregate all of the data into a single MSM to study the average effective dynamics and compare to the concentration dependent dynamics. Other applications include trajectory generation, transition path theory calculations, entropy production estimates, and free-energy analysis. 

Contains methods for processing of sampling data to efficiently compare sampled values to the MSM estimates, as well as caching to enable fast re-construction of models with tweaked parameters. 

Note: this code is intended to be run on systems that are simulated in HOOMD and analyzed using the clustering from my analysis library, found here: https://github.com/onehalfatsquared/SAASH. There are dependencies on the SAASH library within the MultiMSM library, so both are required for now. 

## Package Install Instructions
### Local Machine

If you are installing to a local machine, first clone into the repo. From the highest MultiMSM directory (that contains setup.py), run the following commands:


`python setup.py build`

`python setup.py install`




### Cluster Install

If the above approach does not work, or you are installing to somewhere where you do not have sudo priveleges (like a compute cluster), an alternative approach has been to do the following:

`python setup.py bdist_wheel`

`pip install dist/MultiMSM-0.1.0-py3-none-any.whl --force-reinstall --user`

I think this approach requires the wheel and twine packages. The exact filename inside the dist folder will change with version, so just look for the .whl file created when running the setup script. 

Note that none of the MultiMSM functionality uses parallelization beyond the standard numpy/scipy linear algebra libraries, so running on a cluster is not needed. 
