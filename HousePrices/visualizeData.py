#!/usr/bin/env python3

# import HousePrices.readData
from HousePrices.readData import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def printInfo(data):
    data.info()


def printMissingData(data):
    cols = data.shape[1]
    missingData = data.isnull().sum() > 0
    missingDataCnt = missingData.sum()
    print('Missing values found in ' + str(missingDataCnt) + ' of ' + str(cols) + ' columns')

    columnNames = list(data.columns.values)
    for i in range(cols):
        if missingData[i]:
            print('Column ' + str(i) + ": " + columnNames[i])


def printCorrelations(data):
    c = data.corr()

    # Show heat map
    sns.heatmap(c, xticklabels=1)

    # Correlation with SalePrice
    corrSalePrice = pd.Series(c.SalePrice)
    corrSalePrice = corrSalePrice.abs()
    corrSalePrice.sort_values(inplace=True, ascending=False)

    print(corrSalePrice.nlargest(n=20))


def visualize(data):
    ax = plt.hist(data.SalePrice, bins=25)
    plt.show()
    return 0


# trainData = readData(True)
# testData = readData(False)
trainData = readDataPandas(True)
testData = readDataPandas(False)

# printInfo(trainData)
# printInfo(testData)

printMissingData(trainData)
printCorrelations(trainData)
visualize(trainData)

pass