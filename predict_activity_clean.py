# -*- coding: utf-8 -*-

# predict human activity after reading and cleaning data
# use KFold cross validation

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from read_clean_data import readRawColumns, readRawData, \
    removeDuplicateColumns, rfFitScore, getImportantColumns

print("ya haa")

dfcol, dups = readRawColumns()
dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
dftrain, dftest = removeDuplicateColumns(dfcol, dups, dftrain, dftest)

clf = RandomForestClassifier(n_estimators=10)
score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
impcol = getImportantColumns(dfcol, imp, 0.01)

print("yoo hoo")
