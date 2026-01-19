This folder contains all code for the paper "Bayesian Inference for Partially Observed McKean-Vlasov SDEs with Full Distribution Dependence"

Data Files ‘d3_obs_T_50N.txt’ contains the data for the 3d neuron model, which is required by ‘d3_pmmh_3sigma.m’ and ‘d3_mlpmmh_3sigma.m’.

Main Implementation Files

‘d3_pmmh_3sigma.m’ and ‘3_mlpmmh_3sigma.m’ implement the PMCMC and MLPMCMC algorithms for the 3d neuron model, generating results shown in Figure 1,2,3

For the rates of 3d neuron model for pmmh, run the code 'pmmh_l6.m' on ibex(super computer in KAUST) for different levels (maybe 5-8) each with 64 runs and then calculate the mean at each level and then calculate the rates for pmmh. Similar using the 'mlpmmh_l6.m'(5-8) for mlpmmh rates.(Table 2)
