nndescriptors-epfl
==================

A siamese neural network can learn to discriminate between like/unlike pairs of images


Folders
=======
See comments in each file for further details.

common
% Some common functions such as data structure converters, math and plotting

DeepLearnToolbox
% External toolbox with lots of ANN functionality. Most subprojects use this toolbox.
% DeepLearnToolbox/CNN/cnntrain.m is modified to accept some extra parameters to enable siamese use.

cnn
% Code that runs and plots benchmarking tests for CNN network parameters

nn
% Code that runs and plots benchmarking tests for NN network parameters

nn/own_implementation
% Oldest code in project, from-scratch implementation of a NN

siamese
% The newest part of the project, attempt to make a siamese network.

results
% Data files containing results from benchmarks (which usually take a long time to run)