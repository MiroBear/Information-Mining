from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from StreetSigns.readTrafficSigns import DataReader
from sklearn.model_selection import train_test_split
import numpy as np


def classify(images, labels):
    # images = np.random.random((1000, 100))
    # labels = np.random.randint(10, size=(1000, 1))

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

    print('Training with basic NN')
    # Here is the Sequential model:
    model = Sequential()

    # Stacking layers is as easy as .add():
    dim = np.shape(X_train)[1]
    nClasses = len(np.unique(labels))

    model.add(Dense(units=64, input_dim=dim))  # input_shape=(40, 40)
    model.add(Activation('relu'))
    model.add(Dense(units=nClasses))
    model.add(Activation('softmax'))

    # Once your model looks good, configure its learning process with .compile():
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    y_train_oneHot = to_categorical(y_train, num_classes=nClasses)
    model.fit(X_train, y_train_oneHot, epochs=25, batch_size=32)

    # model.fit(X_train, y_train, epochs=5, batch_size=32)
    y_test_oneHot = to_categorical(y_test, num_classes=nClasses)
    loss_and_metrics = model.evaluate(X_test, y_test_oneHot, batch_size=128)
    print(loss_and_metrics)

    classes = model.predict(X_test, batch_size=128)

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

    images, labels = r.readImages()
    print('#images: ' + str(len(images)))
    classes = np.unique(labels)
    print(np.sort(classes.astype(int)))

    n = len(images)
    imageShape = np.shape(images)
    imageSize = imageShape[1] * imageShape[2]
    features = np.reshape(images, (n, imageSize))
    classify(features, labels)


if __name__ == '__main__':
    main()
