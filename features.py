

import numpy as np
from sklearn.model_selection import train_test_split
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import random
from sklearn.metrics import classification_report
from skimage.filters import gabor_kernel
from skimage.transform import resize
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion
import pickle
from time import time


def ellipseFit(image):
    px = [i for i in range(len(image[0]))]
    py = [i for i in range(len(image))]
    xc = np.dot(np.sum(image, axis=0), px) / np.sum(image)
    yc = np.dot(np.sum(image, axis=1), py) / np.sum(image)
    xvar = np.sum(np.multiply(np.multiply(image, px-xc), px-xc)) / np.sum(image)
    yvar = np.sum(np.multiply(np.multiply(np.transpose(image), py-yc), py-yc)) / np.sum(image)
    xyvar = np.sum(np.multiply(np.transpose(np.multiply(image, px-xc)), py-yc)) / np.sum(image)
    D = np.sqrt((xvar-yvar)*(xvar-yvar) + 4*xyvar*xyvar)
    eig1 = np.sqrt(0.5 * (xvar + yvar + D))
    eig2 = np.sqrt(0.5 * (xvar + yvar - D))
    theta = 0.5 * np.arctan(2*xyvar / (xvar-yvar))
    ecc = np.sqrt(1-(eig2/eig1)**2)
    return xc, yc, eig1, eig2, D, theta, ecc


def gaborTextures(image):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        kernel = np.real(gabor_kernel(0.5, theta=theta,
                                      sigma_x=1, sigma_y=1))
        kernels.append(kernel)
    feats = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats.append(filtered.mean())
        feats.append(filtered.var())
    return feats


ind_voxel =  # pseudocode to represent the individual 28x28x8 potential bacteria voxel
zcompress = np.amax(ind_voxel, axis=0)
xcompress = np.amax(ind_voxel, axis=1)
zthresh = threshold_otsu(zcompress)
zbinary = zcompress > zthresh
zbinary2 = binary_erosion(zbinary)
zbinary3 = binary_erosion(zbinary2)
xthresh = threshold_otsu(zcompress)
xbinary = zcompress > xthresh
ythresh = threshold_otsu(zcompress)
ybinary = zcompress > ythresh
xc, yc, eig1, eig2, D, theta, ecc = ellipseFit(zcompress)
xc2, yc2, eig12, eig22, D2, theta2, ecc2 = ellipseFit(xcompress)
train_features = [np.sum(zbinary), np.sum(zbinary2), np.sum(zbinary3), np.sum(xbinary), np.sum(ybinary)] + \
                 gaborTextures(xcompress) + [xc, yc, eig1, eig2, D, theta, ecc] + \
                 [xc2, yc2, eig12, eig22, D2, theta2, ecc2] + \
                 [np.mean(ind_voxel), np.amin(ind_voxel), np.amax(ind_voxel), np.std(ind_voxel)]