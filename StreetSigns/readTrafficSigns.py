import csv
import matplotlib.pyplot as plt
import numpy as np
import os.path
from hogFeatures import HOGFeatures
import skimage
from skimage.transform import resize


class DataReader(object):
    imagePath = '/home/mirco/PycharmProjects/GTSRB/Training'
    hogPath = '/home/mirco/PycharmProjects/GTSRB_Features_HOG/training/HOG_0'
    huePath = '/home/mirco/PycharmProjects/GTSRB_Features_HueHist/Training/HueHist'

    tmpFileImages = imagePath + '/images.npy'
    tmpFileLabels = imagePath + '/labels.npy'
    tmpFileHOG01 = hogPath + '1' + '/hog01.npy'
    tmpFileHue = huePath + '/hue.npy'
    tmpFileFileNames = imagePath + '/imageFileNames.npy'

    def readImages(self):
        tmpFilesExist = os.path.isfile(self.tmpFileImages) and os.path.isfile(self.tmpFileLabels) and os.path.isfile(self.tmpFileFileNames)
        if tmpFilesExist:
            images = np.load(self.tmpFileImages)
            labels = np.load(self.tmpFileLabels)
            return images, labels

        labels, fileNames = self.readLabels()
        images = self.readImagesIntern(labels, fileNames)

        np.save(self.tmpFileImages, images)
        np.save(self.tmpFileLabels, labels)
        np.save(self.tmpFileFileNames, fileNames)

        return images, labels

    def readHOG(self):
        tmpFilesExist = os.path.isfile(self.tmpFileHOG01) and os.path.isfile(self.tmpFileLabels)
        if tmpFilesExist:
            hogFeatures = np.load(self.tmpFileHOG01)
            labels = np.load(self.tmpFileLabels)
            return hogFeatures, labels

        labels, fileNames = self.readLabels()

        # hogFeaturesProvided = self.readHOGIntern(1, [labels[0]], [fileNames[0]])
        # hogFeatures = self.extractHOG([labels[0]], [fileNames[0]])

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # ax1.hist(hogFeaturesProvided[0], bins=25)
        # ax2.hist(hogFeatures[0], bins=25)
        # plt.show()

        # hogFeatures = self.extractHOG(labels, fileNames)
        hogFeatures = self.readHOGIntern(2, labels, fileNames)

        np.save(self.tmpFileHOG01, hogFeatures)
        np.save(self.tmpFileLabels, labels)
        np.save(self.tmpFileFileNames, fileNames)

        return hogFeatures, labels

    def readHue(self):
        tmpFilesExist = os.path.isfile(self.tmpFileHue) and os.path.isfile(self.tmpFileLabels)
        if tmpFilesExist:
            hueFeatures = np.load(self.tmpFileHue)
            labels = np.load(self.tmpFileLabels)
            return hueFeatures, labels

        labels, fileNames = self.readLabels()

        hueFeatures = self.readHueIntern(labels, fileNames)

        np.save(self.tmpFileHue, hueFeatures)
        np.save(self.tmpFileLabels, labels)
        np.save(self.tmpFileFileNames, fileNames)

        return hueFeatures, labels

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
            # images.append(plt.imread(prefix + fileNames[i]))  # the 1th column is the filename

            image = skimage.io.imread(prefix + fileNames[i], as_grey=True)
            image = resize(image, [40, 40], order=1, mode='reflect')
            images.append(image)  # the 1th column is the filename

        return images

    def extractHOG(self, labels, fileNames):
        n = 1568
        hogFeatures = np.empty([len(labels), n])
        for i in range(0, len(labels)):
            prefix = self.imagePath + '/' + format(int(labels[i]), '05d') + '/'  # subdirectory for class
            imageFileName = prefix + fileNames[i]
            image = HOGFeatures.scaleImage(imageFileName, np.array([40, 40]))
            fd = HOGFeatures.extractFeatures(image)
            hogFeatures[i, :] = fd.reshape([n])

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

    def readHueIntern(self, labels, fileNames):
        hueFeatures = np.empty([len(labels), 256])
        for i in range(0, len(labels)):
            prefix = self.huePath + '/' + format(int(labels[i]), '05d') + '/'  # subdirectory for class
            hueFileName = (prefix + fileNames[i]).replace('.ppm', '.txt')
            hueImage = np.fromfile(hueFileName, sep="\n")
            hueFeatures[i, :] = hueImage

        return hueFeatures

    def clear(self):
        paths = [ self.tmpFileImages, self.tmpFileLabels, self.tmpFileFileNames, self.tmpFileHOG01 ]

        for p in paths:
            if os.path.isfile(p):
                os.remove(p)
