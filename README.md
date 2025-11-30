This folder contains all code for the paper "Bayesian Inference for Partially Observed McKean-Vlasov SDEs with Full Distribution Dependence"

Data Files ‘d3_obs_T_50.txt’ contains the data for the 3d neuron model, which is required by ‘d3_MV.m’ and ‘d3_ml.m’.

Main Implementation Files

‘d3_MV.m’ and ‘d3_ml.m’ implement the PMCMC and MLPMCMC algorithms for the 3d neuron model, generating results shown in Figure 1,2,3,4

For the rates of 3d neuron model for pmmh, run the code 'pmmh_l3_d3_rate.m' on ibex(super computer in KAUST) for different levels (maybe 5-8) each with 64 runs and then calculate the mean at each level and then calculate the rates for pmmh. Similar using the 'mlpmmh_l4_d3_rate.m'(5-8) for mlpmmh rates.(Table 1)
