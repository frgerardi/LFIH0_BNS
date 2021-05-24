## Tutorial

### Environment requirements

The environment.yml is only to be taken as a reference, not all python modules are necessary. Tensorflow 1.15.0 is needed.

### Worflow

The whole set of parameters and settings are inside 'set_params.py'. The structure of the workflow is immediate from the 'main.ipynb' jupyter notebook, from which all the following steps should be run.

1) Simulating a set of mock real observations (the number of sources is currently set to a hundred and selection effects are applied to the distribution of the sources).
2) Generating the set of training and validation sets for training the regression NN used to compress the data down to a set of summary statistics, which dimension is equal to the number of cosmological parameters of interest $\theta$.
3) Generating the set of training data which will be later compressed down and used by pydelfi to learn the conditional $p(t|\theta)$, given $t$ to be the corresponding summary statistics.
4) Training of the regression neural network used for compression
