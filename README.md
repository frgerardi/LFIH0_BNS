# LFIH0_BNS

[![arXiv](https://img.shields.io/badge/arXiv-2104.02728-yellow.svg)](https://arxiv.org/abs/2104.02728)

Code used in arXiv:2104.02728 ("Unbiased likelihood-free inference of the Hubble constant from light standard sirens"), authors: Francesca Gerardi, Stephen M. Feeney, Justin Alsing

### Worflow

The whole set of parameters and settings are inside 'set_params.py'. The structure of the workflow is immediate from the 'main.ipynb' jupyter notebook, from which all the following steps should be run.

1) Simulating a set of mock real observations (the number of sources is currently set to a hundred and selection effects are applied to the distribution of the sources).
2) Generating the set of training and validation sets for training the regression NN used to compress the data down to a set of summary statistics, which dimension is equal to the number of cosmological parameters of interest $\theta$.
3) Generating the set of training data which will be later compressed down and used by pydelfi to learn the conditional $p(t|\theta)$, given $t$ to be the corresponding summary statistics.
4) Training of the regression neural network used for compression (ensemble of neural networks)
5) -- optional -- performing STAN analysis, to later compare pydelfi to standard bayesian inference
6) Run pydelfi, the structure of this procedure is inside 'modules/LFI.py'

### References 

This work is based on pydelfi <a href="https://arxiv.org/abs/1903.00007"> Alsing, Charnock, Feeney and Wandelt 2019</a>, the original code and tutorials can be found at https://github.com/justinalsing/pydelfi. 
