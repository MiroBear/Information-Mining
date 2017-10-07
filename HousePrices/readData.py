#!/usr/bin/env python3

import csv
import pandas as pd


rootDir = '~/PycharmProjects/Kaggle_HousePrices/'
fileTrain = rootDir + 'train.csv'
fileTest = rootDir + 'test.csv'


def readData(isTrainDataRequested):
    data = []

    file = fileTrain if isTrainDataRequested else fileTest
    gtFile = open(file)  # annotations file
    gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
    gtReader.next()  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        data.append(row)
    gtFile.close()

    return data


def readDataPandas(isTrainDataRequested):
    file = fileTrain if isTrainDataRequested else fileTest
    data = pd.read_csv(file)
    return data
