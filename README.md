# Building-CNN-in-Tf-2.0-with-Eager-Execution
This repository hold the building block to create CNN with only lower level Tensorflow 2.0 and Eager Execution (No Keras)
The network uses two blocks of ConvPool Layers followed by two FNN blocks (the number of blocks can be increased as desired).
This CNN was tested on mnist dataset achieving testing Accuracy of 97%. 

in the code there are two functions needed to adjusted in order to configure the CNN
1- weight_baias_init(n_classes) ==> this can be done by adding or removing from the "weights" and "baises" dictionaries.

2- conv_model(X,W,b, dropout=0.5) ==> this can be done by adding or removing layers from the conv_model.
