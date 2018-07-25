# 3D convolutional neural network for identification of bacteria from 3D fluorescence images

This code uses Tensorflow to create three dimensional convolutional neural network for binary classification of bacteria from noise.
It takes in 28x28x8 pixel images and outputs a class of 1 or 0 for bacteria or not bacteria respectively. 

The program inputs the npz file containing the images, performs a train test split on those images to generate the 
train and test set, trains the convolutional neural network and finally tests the network outputing the accuracy.

# Example images
The images were taken by the Parthasarathy lab at the University of Oregon using light sheet microscopy and are of objects detected from 3D image stacks of larval zebrafish. Sloppy segmentation of the gut followed by a difference of gaussians blob detection was used to generate the images. The images are classified as either bacteria, or not bacteria. 

# How to use the code
The code requires the python packages tensorflow, sklearn, numpy and matplotlib to be installed. It is also necessary to make sure
that the path to the npz file containing the images, the variable called path_to_images in the code, is correct on your machine. 
After that, the code should be usable as is. See Tensorflow's documentation to save a trained network. 

# How to edit the code
The hyperparameters are initialized next to one another for easy editing. Adding new convolutional, pooling or dense layers is as 
simple as calling the appropriate functions in the network architecture sections, making sure that the following correctly 
referencing the added layer and that the number of inputs and outputs (num_in, num_out) are consistent.

# Notes on the images and the npz file
Note that the pixel dimensions of each image are 28x28x8, each image is normalized such that they have a standard deviation of 1 and zero mean. The labels are in one-hot encoding. The images and labels are stored in a numpy array as \[data, labels\]. 
