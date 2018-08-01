



import numpy as np
from skimage.feature import blob_dog
from skimage.measure import block_reduce
from skimage import exposure
from time import time
from scipy import ndimage
import glob
import re



def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def dist(x1, y1, list):
    x2 = list[0]
    y2 = list[1]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def difference_of_gaussians_2D(file_names, scale, min_sig=2, max_sig=20, thrsh=0.02):
    global xpixlength
    global ypixlength
    plots = []
    start_time = time()
    blobs = []
    print('starting loop')
    pix_dimage = ndimage.imread(file_names[0], flatten=True)
    ypixlength = len(pix_dimage[0])
    xpixlength = len(pix_dimage)
    for name in file_names:
        image = ndimage.imread(name, flatten=True)
        image = block_reduce(image, block_size=(scale, scale), func=np.mean)
        plots.append(image.tolist())
        image = (image - np.min(image))/np.max(image)
        tempblobs = blob_dog(image, max_sigma=max_sig, min_sigma=min_sig, threshold=thrsh, overlap=0).tolist()
        for tempblob in tempblobs:
            tempblob.append(0)
            tempblob.append(0)
        if tempblobs == []:
            blobs.append([[]])
        else:
            blobs.append(tempblobs)
        print(name)
    return blobs, plots


def segmentation_mask(plots1, wdth, thresh2):
    plots_out = [[] for el in range(len(plots1))]
    plots2 = [[] for el in range(len(plots1))]
    print('building mask...')
    for i in range(len(plots1)):
        image = plots1[i]
        image = (image - np.min(image))/np.max(image)
        plots2[i] = exposure.equalize_hist(np.array(image))
    for i in range(len(plots2)):
        if i < int(wdth/2):
            image = np.mean(plots2[0: wdth], axis=0)
        elif i > int(len(plots2) - wdth/2):
            image = np.mean(plots2[-wdth: -1], axis=0)
        else:
            image = np.mean(plots2[i-int(wdth/2):i+int(wdth/2)], axis=0)
        binary = image > thresh2
        plots_out[i] = binary
    return plots_out


def trim_segmented(blobs, plots, wdth=30, thresh2=0.7):
    global trim_time
    plots1 = segmentation_mask(plots, wdth, thresh2)
    trim_time = time()
    print('done building the mask')
    for z in range(len(blobs)):
        rem = []
        for blob in blobs[z]:
            if plots1[z][int(blob[0])][int(blob[1])] is False and blob != []:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs
#                                Loop through blobs trimming consecutive blobs

def trim_consecutively(blobs, adjSize=2):
    for z in range(len(blobs)):
        for n in range(len(blobs[z])):
            if blobs[z][n][2] == 0:
                break
            else:
                blobs[z][n][2] = 1
                contains = 'True'
                zz = z + 1
                testlocation = blobs[z][n][0:2]
                while contains == 'True' and zz < len(blobs):
                    if blobs[zz] == []: #  check for empty zz
                        break
                    for blob in blobs[zz]:
                        if dist(blob[0], blob[1], testlocation) < adjSize:
                            blobs[z][n][2] += 1
                            testlocation = blob[0:2]
                            # x-end
                            blobs[z][n][3] = testlocation[0]
                            # x-end
                            blobs[z][n][4] = testlocation[1]

                            blobs[zz].remove(blob)
                            zz += 1
                            contains = 'True'
                            break
                        else:
                            contains = 'False'
    return blobs


#                            trim when blob only in one or two planes
def trim_toofewtoomany(blobs, tooFew=2, tooMany=15):
    for z in range(len(blobs)):
        rem = []    # note, removing while looping skips every other entry to be removed
        for blob in blobs[z]:
            if blob[2] < tooFew or blob[2] > tooMany:
                rem.append(blob)
            # the following makes sure blobs aren't on x-y edge of image
            elif blob[0] < cube_length or blob[1] < cube_length:
                rem.append(blob)
            elif blob[0] > xpixlength - cube_length:
                rem.append(blob)
            elif blob[1] > ypixlength - cube_length:
                rem.append(blob)
        for item in rem:
            blobs[z].remove(item)
    return blobs


def cubeExtractor(extracted_ROI):
    z = 0
    cubes = [[] for i in extracted_ROI]
    for name in file_names:
        z += 1
        image = ndimage.imread(name, flatten=True)  # CHANGE TO EXTRACT FROM PLOTS
        for el in range(len(extracted_ROI)):
            if extracted_ROI[el][2] > len(blobs) - int(z_length / 2) and z > len(blobs) - z_length:
                x_start = int(extracted_ROI[el][0] - cube_length / 2)
                y_start = int(extracted_ROI[el][1] - cube_length / 2)
                subimage = image[x_start:x_start + cube_length, y_start:y_start + cube_length].tolist()
                cubes[el].append(subimage)
            elif extracted_ROI[el][2] > z + int(z_length / 2):
                break
            elif extracted_ROI[el][2] <= int(z_length / 2) and z <= z_length:
                x_start = int(extracted_ROI[el][0] - cube_length / 2)
                y_start = int(extracted_ROI[el][1] - cube_length / 2)
                subimage = image[x_start:x_start + cube_length, y_start:y_start + cube_length].tolist()
                cubes[el].append(subimage)
            elif extracted_ROI[el][2] > z - int(z_length / 2):
                x_start = int(extracted_ROI[el][0] - cube_length / 2)
                y_start = int(extracted_ROI[el][1] - cube_length / 2)
                subimage = image[x_start:x_start + cube_length, y_start:y_start + cube_length].tolist()
                cubes[el].append(subimage)
    return cubes



file_names = glob('file_directory_to/example_vibrio_bulb_images/*.png')
scale = 4
cube_length = 28
z_length = 6

blobs, plots = difference_of_gaussians_2D(file_names, scale)


########################################################################################################################
#                                           TRIMMING                                                                   #

blobs = trim_segmented(blobs, plots)  # remove detected objects away from crude approximation of the gut
blobs = trim_consecutively(blobs)  # stitch together detected objects along the z-dimension
blobs = trim_toofewtoomany(blobs)  # remove blobs that are two short or long in z

#  blibs is one-d list of (x,y,z, bacType) for detected blobs
ROI_locs = [[blobs[i][n][0] * scale + (blobs[i][n][3] - blobs[i][n][0]) / 2 * scale,
             blobs[i][n][1] * scale + (blobs[i][n][4] - blobs[i][n][1]) / 2 * scale,
             int(i + blobs[i][n][2] / 2)] for i in range(len(blobs)) for n in range(len(blobs[i]))]
ROI_locs = sorted(ROI_locs, key=lambda x: x[2])

########################################################################################################################
#                                           CUBE EXTRACTOR                                                             #
#                          ( extract a cube around each blob for classification )                                      #
#                          ( cubes is indexed by blob, z, x,y )                                                        #

cubes = cubeExtractor(ROI_locs)

