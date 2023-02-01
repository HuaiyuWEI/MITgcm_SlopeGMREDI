# Overview

Energetically-constrained, bathymetry-aware parameterizations of buoyancy diffusivity ([Wang and Stewart, 2020](https://www.sciencedirect.com/science/article/pii/S1463500319301775?via%3Dihub)) are implemetented into [MITgcm](http://mitgcm.org) (source code checkpoint 68i) on the basis of the [GMREDI package](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/gmredi.html). The parameterizations are augmented by an artificial neural network (ANN) that infers the mesoscale eddy kinetic energy (EKE) from the mean flow and topographic quantities online. Moreover, another ANN is employed to parameterize the cross-slope eddy momentum flux (EMF) that maintains an analogous barotropic flow field to that in an eddy-resolving model.

# Implementation steps
**The energetically-constrained GM variants are implemented in MITgcm by taking the following coding steps:**

**(1)** Within the GMREDI package of MITgcm, modify the initialization subroutine *gmredi\_readparms.F* to read in the neural network, which is saved as arrays upon the completion of training procedure.
    
**(2)** Within the GMREDI package, code up a new subroutine *gmredi\_calc\_ml.F* that calculates the ANN input variables and infers the EKE at each model time step.
    
**(3)** Within the subroutine of *gmredi\_calc\_ml.F*, formulate the energetically-constrained diffusivity variants, using the ANN-inferred EKE and the MITgcm-simulated quantities at each model time step.
    
**(4)** Call the subroutine of *gmredi\_calc\_ml.F* within *gmredi\_calc\_tensor.F* to formulate the eddy diffusivity tensor associated with the energetically-constrained diffusivity variants, which then informs the parameterized eddy fluxes incorporated into the tracer equation in MITgcm. 


**The EMF is parameterized following three major steps:**

**(1)** Read the neural network for inferring the EMF via the subroutine of *gmredi\_readparms.F*, and infer the EMF within the subroutine of *gmredi\_calc\_ml.F* (note that this neural network employs identical input quantities as that for inferring the EKE) at each model time step. The ANN-inferred EMF are then stored as global variables in MITgcm.
    
**(2)** Modify the subroutine of *calc\_eddy\_stress.F* to calculate the depth-averaged divergence of EMF, and call *calc\_eddy\_stress.F* within the subroutine of *dynamics.F* at each model time step.  

**(3)** Within the subroutine of *dynamics.F*, incorporate the depth-averaged EMF divergence into the momentum equation of MITgcm as an external body forcing, realized via the subroutines of *timestep.F* and *apply\_forcing.F*.
