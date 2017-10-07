#!/usr/bin/env python3

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize


class HOGFeatures(object):
    @staticmethod
    def scaleImage(imageFileName, tgtSize):
        im = imread(imageFileName, as_grey=True)
        imOut = resize(im, tgtSize, order=1, mode='reflect')

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # ax1.axis('off')
        # ax1.imshow(im, cmap=plt.cm.gray)
        # ax2.axis('off')
        # ax2.imshow(imOut, cmap=plt.cm.gray)
        # plt.show()

        return imOut

    @staticmethod
    def extractFeatures(image):
        o = 8
        pc = (5, 5)
        cb = (2, 2)
        bn = 'L2-Hys'
        fv = True
        ts = True
        vis = False

        if vis:
            fd, hog_image\
                = hog(image, orientations=o, pixels_per_cell=pc,
                      cells_per_block=cb, block_norm=bn, visualise=vis, feature_vector=fv, transform_sqrt=ts)
        else:
            fd\
                = hog(image, orientations=o, pixels_per_cell=pc,
                      cells_per_block=cb, block_norm=bn, visualise=vis, feature_vector=fv, transform_sqrt=ts)

        if vis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # sharex=True, sharey=True)

            ax1.axis('off')
            ax1.imshow(image, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            ax1.set_adjustable('box-forced')

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            ax1.set_adjustable('box-forced')
            plt.show()

        if vis:
            return fd, hog_image
        else:
            return fd
