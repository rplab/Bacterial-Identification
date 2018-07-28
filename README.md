# 3D convolutional neural network for identification of bacteria from 3D fluorescence images

Included is code for the convolutional neural network used, ConvNet_3D_for_bacterial_identificaiton.py, as well as for extracting the 28x28x8 pixel images.
The convnet code uses Tensorflow to create a 3D convolutional neural network for binary classification of bacteria from noise.
It takes in 28x28x8 pixel voxel images and outputs a class of 1 or 0 for bacteria or not bacteria respectively. 

The program inputs the npz file containing the images and labels, performs a train test split on those images to generate the 
train and test set, trains the convolutional neural network and finally tests the network outputing the accuracy.

# Example voxel images
The images were taken by the Parthasarathy lab at the University of Oregon using light sheet microscopy and are of objects detected from 3D image stacks of larval zebrafish. Sloppy segmentation of the gut followed by a difference of gaussians blob detection was used to generate the images. The images are classified as either bacteria, or not bacteria. Note that the data and labels are available at www.dropbox.com/sh/pi814beai6sihaw/AACdfBkajyOwz4s9jV3vW57ua?dl=0.

# How to use the code
The code requires the python packages tensorflow, sklearn, numpy and matplotlib to be installed. It is also necessary to make sure
that the path to the npz file containing the images, the variable called "file_loc" in the code, is correct on your machine. 
After that, the code should be usable as is. See Tensorflow's documentation to save a trained network. 

# Notes on the images and the npz file
Note that the pixel dimensions of each image are 28x28x8, each image is normalized such that they have a standard deviation of 1 and zero mean. The labels should be converted to one-hot encoding. The images and labels are stored in a npz file. See Dropbox link for details.
