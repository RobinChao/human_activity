# -*- coding: utf-8 -*-

# predict human activity after reading and cleaning data
# use KFold cross validation

#import pandas as pd
#from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

from read_clean_data import readRawColumns, readRawData, \
    removeDuplicateColumns, rfFitScore, getImportantColumns, \
    getPlotDir, plotImportances

kfolds = 4
print("random forest, kfold", kfolds)

dfcol, dups = readRawColumns()
dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
dftrain, dftest = removeDuplicateColumns(dfcol, dups, dftrain, dftest)

# separate dftrain into train, validate

clf = RandomForestClassifier(n_estimators=100)

print("dftrain shape", dftrain.shape)

def summit(impsum, imp):
    '''add importances (basically set intersection)'''
    if len(impsum) == 0:
        impsum = imp
    else:
        impsum = [ x+y for x,y in zip(impsum, imp) ]
    return impsum

def normImportances(imps, kinv):
    '''normalize importances by kfold'''
    return list(map(lambda e: e * kinv, imps))

total_score = 0
imps = []
kf = KFold(dftrain.shape[0], n_folds=kfolds)
for train, test in kf:
    subtrain, subtest, subtrain_y, subtest_y = dftrain.iloc[train], \
      dftrain.iloc[test], dftrain_y.iloc[train], dftrain_y.iloc[test]
    score, imp = rfFitScore(clf, subtrain, subtrain_y,subtest, subtest_y)
    total_score += score
    imps = summit(imps, imp)

kinv = 1.0 / kfolds
total_score *= kinv
print("total score", total_score)

#clf = RandomForestClassifier(n_estimators=10)
#score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
imps = normImportances(imps, kinv)
impcol = getImportantColumns(dftrain.columns, imps)
print("importance column len", len(impcol))
print("KFold: top ten important scores and features:\n", impcol[:10])

plotdir = getPlotDir()
plotImportances(impcol, plotdir, "kfold")

# Q: how to combine folds into one model for prediction?
#    usually clfit.predict( dftest )
#    use sum of importances, top twenty?

