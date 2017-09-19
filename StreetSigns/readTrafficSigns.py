# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path
from hogFeatures import HOGFeatures


class DataReader(object):
    imagePath = '/home/mirco/PycharmProjects/GTSRB/Training'
    hogPath = '/home/mirco/PycharmProjects/GTSRB_Features_HOG/training/HOG_0'

    tmpFileImages = imagePath + '/images.npy'
    tmpFileLabels = imagePath + '/labels.npy'
    tmpFileHOG01 = hogPath + '1' + '/hog01.npy'
    tmpFileFileNames = imagePath + '/imageFileNames.npy'

    def readImages(self):
        tmpFilesExist = os.path.isfile(self.tmpFileImages) and os.path.isfile(self.tmpFileLabels) and os.path.isfile(self.tmpFileFileNames)
        if tmpFilesExist:
            images = np.load(self.tmpFileImages)
            labels = np.load(self.tmpFileLabels)
            fileNames = np.load(self.tmpFileFileNames)
            return images, labels

        labels, fileNames = self.readLabels()
        images = self.readImagesIntern(labels, fileNames)

        np.save(self.tmpFileImages, images)
        np.save(self.tmpFileLabels, labels)
        np.save(self.tmpFileFileNames, fileNames)

        return images, labels

    def readHOG(self, hogType):
        tmpFilesExist = os.path.isfile(self.tmpFileHOG01) and os.path.isfile(self.tmpFileLabels)
        if tmpFilesExist:
            hogFeatures = np.load(self.tmpFileHOG01)
            labels = np.load(self.tmpFileLabels)
            fileNames = np.load(self.tmpFileFileNames)

            return hogFeatures, labels

        labels, fileNames = self.readLabels()
        hogFeatures = self.readHOGIntern(1, labels, fileNames)

        np.save(self.tmpFileHOG01, hogFeatures)
        np.save(self.tmpFileLabels, labels)
        np.save(self.tmpFileFileNames, fileNames)

        return hogFeatures, labels

    def readLabels(self):
        labels = []  # corresponding labels
        fileNames = []
        # loop over all 42 classes
        for c in range(0, 43):
            prefix = self.imagePath + '/' + format(c, '05d') + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            gtReader.next()  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                labels.append(row[7])  # the 8th column is the label
                fileNames.append(row[0])
            gtFile.close()
        return labels, fileNames

    # function for reading the images
    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of images, list of corresponding labels
    def readImagesIntern(self, labels, fileNames):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        images = []  # images
        # loop over all 42 classes
        for i in range(0, len(labels)):
            prefix = self.imagePath + '/' + format(int(labels[i]), '05d') + '/'  # subdirectory for class
            images.append(plt.imread(prefix + fileNames[i]))  # the 1th column is the filename

        return images

    def extractHOG(self, labels, fileNames):
        # TODO: extract HOG features using HOGFeatures class
        # TODO: image should be scaled as well!
        hogFeatures = np.empty([len(labels), 1568])
        for i in range(0, len(labels)):
            prefix = self.imagePath + '/' + format(int(labels[i]), '05d') + '/'  # subdirectory for class
            imageFileName = prefix + fileNames[i]
            image = plt.imread(imageFileName)
            fd = HOGFeatures.extractFeatures(image)
            hogFeatures[i, :] = fd

        return hogFeatures

    def readHOGIntern(self, hogType, labels, fileNames):
        hogPathWithType = self.hogPath + str(hogType)
        hogFeatures = np.empty([len(labels), 1568])
        for i in range(0, len(labels)):
            prefix = hogPathWithType + '/' + format(int(labels[i]), '05d') + '/'  # subdirectory for class
            hogFileName = (prefix + fileNames[i]).replace('.ppm', '.txt')
            hogImage = np.fromfile(hogFileName, sep="\n")
            hogFeatures[i, :] = hogImage

        return hogFeatures

    def clear(self):
        os.remove(self.tmpFileImages)
        os.remove(self.tmpFileLabels)
        os.remove(self.tmpFileHOG01)
        os.remove(self.tmpFileFileNames)
