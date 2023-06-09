# NO3RR_GMPNP_2023

This script is an adaptation of Divya Bohra's GMPNP model for CO2R, as referenced below. Please follow Divya's instructions for installing and using the model copied below, or at the [original repo](https://github.com/divyabohra/GMPNP). Additionally, if you are planning to use both the PNP and GMPNP approaches, you are recommended to start with Divya's code as the NO3RR code was *not* modified for PNP simulations. 

The environment files have been modified for Python 3, but further modification may be necessary for future users. If you found the NO3RR code useful, please cite both our and Divay's work using the following information: 

Guo et al. : (TO BE UPDATED) 

Bohra et al. : 10.5281/zenodo.4106373 or use the following BibTeX entry: 

@software{divya_bohra_2020,
  author       = {Divya Bohra},
  title        = {{divyabohra/GMPNP: First release of 1D and 3D GMPNP 
                   for CO2 electrocatalysis}},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.4106373},
  url          = {https://doi.org/10.5281/zenodo.4106373}



## Divya's Instructions for 1D and 3D GMPNP Model: 
[![DOI](https://zenodo.org/badge/295436920.svg)](https://zenodo.org/badge/latestdoi/295436920)

This repository contains the implementation of generalised modified Poisson-Nernst-Planck (**GMPNP**) for a CO2 electrocatalysis system using the FEniCS finite element package in Python. The implementation is available for 

1. a 1D system with CO2 dissolved in electrolyte (H-cell type) and 
2. a 3D cylindrical catalytic pore for a GDE-based system.

The parameter and mesh files for both the 1D and 3D cases can be found in the utilities folder. 
The script for the Stern layer solves the Poisson equation in the Stern region using the results obtained from the GMPNP model. 
Additionally, scripts to solve the reaction diffusion system for both 1D and 3D cases are available for comparison.

The 1D scripts were used to derive the results published in [https://pubs.rsc.org/en/content/articlelanding/2019/ee/c9ee02485a#!divAbstract](url).

The environment.yml file can be used with conda to create the python environment needed to run the script using the command:

`conda env create -f environment.yml`

The environment name is the first line in the yaml file and is currently set as **FEniCS**.

Once the environment is created, activate it using:

`conda activate FEniCS`

To see the packages installed in the new environment, use:

`conda env list`

See [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#](url) for further help regarding managing enviroments using conda. 

In order to run the scripts (both 1D and 3D), please check the ***basepath*** and ***basepath_utilities*** variables according to the location of the python file and the utilities folder on your local machine. 

Activate the FEniCS virtual environment and run the MPNP script from the "1D" directory (as an example) using:

`python MPNP_CO2ER_EDL.py`

As you will see in the scripts, you can vary all relevant parameters using command line arguements like in the example below. This makes it simpler to submit multiple jobs on a computational cluster.

`python MPNP_CO2ER_EDL.py --voltage_multiplier=-10.0 --cation='Cs'`

The 3D pore scripts can be run following a similar procedure.

And you are good to go!

If you find this code useful, please **cite** it using the DOI: 10.5281/zenodo.4106373 or use the following BibTeX entry: 

@software{divya_bohra_2020,
  author       = {Divya Bohra},
  title        = {{divyabohra/GMPNP: First release of 1D and 3D GMPNP 
                   for CO2 electrocatalysis}},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.4106373},
  url          = {https://doi.org/10.5281/zenodo.4106373}
}
