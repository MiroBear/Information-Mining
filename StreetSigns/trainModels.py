#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.preprocessing import normalize

from StreetSigns.readTrafficSigns import DataReader


def classify(images, labels):
    # create stratified labels with equal number of samples per class
    classes, classCnts = np.unique(labels, return_counts=True)
    evenCnt = len(labels) / len(classes)
    evenCnt = int(evenCnt)
    stratifiedLabels = np.repeat(classes, evenCnt)
    stratifiedLabels = np.resize(stratifiedLabels, len(labels))
    remainder = len(labels) - evenCnt * len(classes)
    stratifiedLabels[len(labels)-remainder-1:-1] = classes[0:remainder]

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.25, random_state=0, stratify=stratifiedLabels)  # labels)

    # dimRed = LDA()
    # X_train = dimRed.fit_transform(X_train, y_train)
    # X_test = dimRed.transform(X_test)

    print('Training with LinearSVC')
    clf = svm.LinearSVC(penalty='l2', C=0.5)
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    print(scores)

    # TODO: reduce dimensionality of features by PCA and do classification on this data. This should get rid of
    # background noise.

    # TODO: use spatial pyramid matching
    # https://github.com/wihoho/Image-Recognition

    # TODO: get best individuals from each class, extract the shape and create a mask to cancel out the background
    # for all images of this class.

    # y_pred = clf.predict(X_test)
    # confSVC = confusion_matrix(y_test, y_pred)
    # plt.figure()
    # plot_confusion_matrix(confSVC, classes=classes)
    # plt.show()

    # print('Training with LDA')
    # clf = LDA(solver='lsqr', shrinkage='auto')
    # clf.fit(X_train, y_train)
    # scores = clf.score(X_test, y_test)
    # print(scores)

    # print('Training with LR')
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # scores = clf.score(X_test, y_test)
    # print(scores)
    #
    # print('Training with Random Forest')
    # clf = RandomForestClassifier(n_estimators=100)
    # clf.fit(X_train, y_train)
    # scores = clf.score(X_test, y_test)
    # print(scores)


def main():
    r = DataReader()
    # r = reader.DataReader()

    # r.clear()

    images, labels = r.readImages()
    print('#images: ' + str(len(images)))
    classes = np.unique(labels)
    print(np.sort(classes.astype(int)))

    hogFeatures, labels = r.readHOG()
    print('#hogFeatures: ' + str(len(hogFeatures[0])))

    hueFeatures, labels = r.readHue()
    print('#hueFeatures: ' + str(len(hueFeatures[0])))

    features = combine(hogFeatures, hueFeatures)
    # features = hogFeatures

    # imagesVectorized = np.empty([len(labels), images[0].size])
    # for i in range(len(labels)):
    #     imagesVectorized[i] = images[i].ravel()
    # features = imagesVectorized

    # features = normalize(features)
    # HOG1
    # no normalize: 0.982432432432
    # normalize:    0.980630630631

    # HOG2
    # 0.991441441441
    # same with hue features combined

    # np.histogram(labels.astype(int))
    # plt.hist(labels.astype(int), bins=42)
    # plt.title('Classes')
    # plt.show()

    # Classes are not evenly distributed
    # TODO:
    # - modify images a bit to create extra variants, e.g. by scaling, rotating, blurring, distorting
    # - or apply stratified sampling during cross-validation or training-test split

    '''
    TODO: extract main colors from all images as additional feature.
    Problems:
    - common colors for all images, i.e. define a small set of colors to which all are mapped. Could be problematic for
      corner colors.
    - main colors to be extracted e.g. by ColorThief: https://github.com/fengsp/color-thief-py/blob/master/colorthief.py
      -> per image and image dependent. i.e. no mapping to predefined colors. This must be done on our own.    
    '''
    classify(features, labels)


def combine(X1, X2):
    n = len(X1)
    n1 = len(X1[0])
    n2 = len(X2[0])

    X = np.empty([n, n1 + n2])

    for i in range(len(X)):
        X[i, 0:n1] = X1[i]
        X[i, n1:n1+n2] = X2[i]

    return X


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    np.fill_diagonal(cm, 1)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()
