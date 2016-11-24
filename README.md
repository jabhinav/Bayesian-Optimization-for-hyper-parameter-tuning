# Bayesian-Optimization-for-hyper-parameter-tuning
Spearmint integrated Bayesian Optimization for hyper parameter tuning of Auto sparse encoder embedded with Softmax Classifier for MNIST digit Classification.


Instructions to run
------------------------------

1. Download and install Spearmint package (instructions are on 'https://github.com/JasperSnoek/spearmint')
2. Download the MNIST dataset (from http://yann.lecun.com/exdb/mnist/) in the same folder with the rest of Matlab files
3. Run the spearmint optimization module

* Implementation of Classification module is in Matlab.
* STL_opt is the matlab wrapper required for spearmint package.
* config.json is the configuration file with specifications as per the spearmint instruction.

* L-BFGS algorithm is used to minimize the cost function for weights training in Softmax Classifier and Sparse Auto-encoder
