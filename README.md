# MultiMSM

This repository contains a framework for creating a collection of Markov State Models (MSMs) by leveraging data obtained at different values of certain parameters. 

For example, my application is to construct an MSM that models the dynamics of a self-assembling system with a finite pool of subunits. The rates of forming various structures depend on the concentration of free monomers, so transition events are augmented with the monomer fraction present when they occur. The user may provide their own discretization of their variable parameter (monomer fraction for me), and this library will construct multiple MSMs, one for each intrval of the discretization. 

Using the monomer fraction, one can solve the forward and backward Kolmogorov equations for this collection of MSMs. We can also aggregate all of the data into a single MSM to study the average effective dynamics and compare to the concentration dependent dynamics. 

Will also contain methods for processing of sampling data to efficiently compare sampled values to the MSM estimates. 


#Package Install Instructions (For Now)
To be able to import this code as a package outside of the source directory, I have been using a python setup file to make a pip package and install it. It is not published yet (maybe I will do this eventually, after finalizing and documenting some things), but this works for now. 

If you are installing to a local machine, first clone into the repo. From the highest MultiMSM directory (that contains setup.py), run the following commands:

python setup.py build

python setup.py install


This should allow MultiMSM to be imported from anywhere. 

If the above approach does not work, or you are installing to somewhere where you do not have sudo priveleges (like a compute cluster), an alternative approach has been to do the following:

python setup.py bdist_wheel

pip install dist/MultiMSM-0.1.0-py3-none-any.whl --force-reinstall --user

I think this approach requires the wheel and twine packages. The exact filename inside the dist folder may change, so just look for the .whl file when running the command. 
