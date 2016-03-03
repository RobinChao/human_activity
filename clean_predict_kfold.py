# -*- coding: utf-8 -*-

# predict human activity after reading and cleaning data
# use KFold cross validation

#import pandas as pd
#from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

from read_clean_data import readRawColumns, readRawData, \
    removeDuplicateColumns, rfFitScore, getImportantColumns

print("random forest, kfold 4")

dfcol, dups = readRawColumns()
dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
dftrain, dftest = removeDuplicateColumns(dfcol, dups, dftrain, dftest)

clf = RandomForestClassifier(n_estimators=100)

print("dftrain shape", dftrain.shape)

def summit(impsum, imp):
    if len(impsum) == 0:
        impsum = imp
    else:
        impsum = [ x+y for x,y in zip(impsum, imp) ]
    return impsum

total_score = 0
imps = []
kf = KFold(dftrain.shape[0], n_folds=4)
for train, test in kf:
    subtrain, subtest, subtrain_y, subtest_y = dftrain.iloc[train], \
      dftrain.iloc[test], dftrain_y.iloc[train], dftrain_y.iloc[test]
    score, imp = rfFitScore(clf, subtrain, subtrain_y,subtest, subtest_y)
    total_score += score
#    imps.append(imp)
    imps = summit(imps, imp)

total_score *= 0.25
print("total score", total_score)

#imps = list(zip(*imps))
#total_imp = [ sum(imp) for imp in imps ]
#print("imps len", len(imps), imps[:5])  # unsorted

#clf = RandomForestClassifier(n_estimators=10)
#score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
impcol = getImportantColumns(dfcol, imps)
print("importance column len", len(impcol))
print("KFold: top ten important scores and features:\n", impcol[:10])

