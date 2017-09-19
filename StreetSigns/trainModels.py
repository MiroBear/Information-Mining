
import readTrafficSigns as reader
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hogFeatures import HOGFeatures


def classify(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.4, random_state=0)

    print('Training with LinearSVC')
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    print(scores)

    print('Training with LR')
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    print(scores)

    print('Training with Random Forest')
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    print(scores)


def main():
    r = reader.DataReader()

    r.clear()

    images, labels = r.readImages()
    print('#images: ' + str(len(images)))
    classes = np.unique(labels)
    print(np.sort(classes.astype(int)))

    hogFeatures, labels = r.readHOG(1)
    print('#hogFeatures: ' + str(len(hogFeatures)))

    # np.histogram(labels.astype(int))
    # plt.hist(labels.astype(int), bins=42)
    # plt.title('Classes')
    # plt.show()

    # Classes are not evenly distributed
    # TODO: modify images a bit to create extra variants, e.g. by scaling, rotating, blurring, distorting

    '''
    TODO: generate HOG features on your own
    http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
    '''
    fd, hog_image = HOGFeatures.extractFeatures(images[0])

    '''
    TODO: extract main colors from all images as additional feature.
    Problems:
    - common colors for all images, i.e. define a small set of colors to which all are mapped. Could be problematic for
      corner colors.
    - main colors to be extracted e.g. by ColorThief: https://github.com/fengsp/color-thief-py/blob/master/colorthief.py
      -> per image and image dependent. i.e. no mapping to predefined colors. This must be done on our own.    
    '''
    # classify(hogFeatures, labels)


if __name__ == '__main__':
    main()
