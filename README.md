# Convolutional neural network for identification of bacteria from 3D fluorescence images

(Associated with Hay and Parthasarathy, *Performance of convolutional neural networks for identification of bacteria in 3D microscopy datasets* (2018))

## Links to Data

Shared data, including all 21000 manually labeled 3D regions of interest.
- [filename for all extracted Vibrio volumes and labels, as one .npz file]
- [filename for all extracted Pseudomonas volumes, one .npz file]
- readme_data.txt : Information about the above image and label files.
- [filename for images making up a full 3D scan of the anterior "bulb" region of one larval zebrafish]
- [filename for all manual labels, and the CNN classifications, for the "bulb" scan above, as one .npz file]

### Notes on the images and the npz file
Each extracted image is 28x28x8 px (4.5 x 4.5 x 8 um), normalized such that it has a standard deviation of 1 and zero mean. 


## Code

This folder contains:
- ConvNet_3D_for_bacterial_identificaiton.py: code for the convolutional neural network
- ROI_extractor.py : code for extracting the 28x28x8 pixel images.
- features.py : code for extracting features from images (for SVM and RF classifications), and descriptions of features.

The ConvNet code uses Tensorflow to create a 3D convolutional neural network for binary classification of objects as bacteria or noise.
It takes as input in 28x28x8 pixel images and outputs a class label of 1 or 0 for bacteria or not bacteria respectively. The input takes the form of an npz file containing the images and labels, then performs a train / test split on those images to generate the 
train and test sets, trains the convolutional neural network and finally tests the network outputting the test accuracy.

### How to use the code
The code requires the python packages tensorflow, sklearn, numpy and matplotlib to be installed. It is also necessary to make sure
that the path to the npz file containing the images, the variable called "file_loc" in the code, is correct on your machine. 
After that, the code should be usable as is. See Tensorflow's documentation to save a trained network. 

