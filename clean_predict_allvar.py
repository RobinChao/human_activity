# -*- coding: utf-8 -*-

# predict human activity after reading and cleaning data
# use Random Forest w/ all anonymous variable names

#import pandas as pd
import numpy as np
#from functools import reduce
from sklearn.ensemble import RandomForestClassifier

from read_clean_data import readRawColumns, readRawData, rfFitScore, \
    getImportantColumns, getPlotDir, plotImportances

def rename_columns(df):
    '''rename columns x0-xn except subject and activity (latter is Y)'''
    dlen = df.shape[1] - 1  # assume last column stays same
    newcol = list(map(lambda s: "x" + str(s), list(range(dlen))))
    newcol.append(df.columns[-1])
    df.columns = newcol
    return df

# six subjects total, not all subjects in dftrain
def get_validation_data(dfx):
    '''get validation data'''
    return dfx[(dfx['subject'] >= 19) & (dfx['subject'] < 27)]

if __name__ == '__main__':
    dfcol, dups = readRawColumns()
    dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
    
    dftrain = rename_columns(dftrain)
    dftest = rename_columns(dftest)
    print("dftrain shape head", dftrain.shape, "\n", dftrain[:5])
    print("dftest shape head", dftest.shape, "\n", dftest[:5])
    
    # which subjects?
    subjtrain = set(list(dftrain['subject']))
    subjtest = set(list(dftest['subject']))
    print("train subject set ( size", len(subjtrain), ")", subjtrain)
    print("test subject set ( size", len(subjtest), ")", subjtest)
# the thing is, raw train, test data is in separate files, different subjects
# let's do quick and easy way: train>=27, test<=6, validate>=21 <27
#   within current data, don't merge train w/ test

    # add subject to y's before split
    dftrain_y['subject'] = dftrain['subject']
    dftest_y['subject'] = dftest['subject']
    
    dfvalid = get_validation_data(dftrain)     # six subjects total
    dfvalid_y = get_validation_data(dftrain_y)
    dfvalid_y = dfvalid_y.drop('subject', axis=1)
    print("dfvalid shape head", dfvalid.shape, "\n", dfvalid[:5])
    print("dfvalid_y shape head", dfvalid_y.shape, "\n", dfvalid_y[:5])
    
    dftest = dftest[dftest['subject'] < 14]  # six subjects total
    dftest_y = dftest_y[dftest_y['subject'] < 14]
    dftest_y = dftest_y.drop('subject', axis=1)
    print("dftest shape head", dftest.shape, "\n", dftest[:5])
    print("dftest_y shape head", dftest_y.shape, "\n", dftest_y[:5])
    
    dftrain = dftrain[dftrain['subject'] >= 27]  # four subjects total
    dftrain_y = dftrain_y[dftrain_y['subject'] >= 27]
    dftrain_y = dftrain_y.drop('subject', axis=1)
    print("dftrain shape head", dftrain.shape, "\n", dftrain[:5])
    print("dftrain_y shape head", dftrain_y.shape, "\n", dftrain_y[:5])
    
    # really what I want with validation is try several parameters
    scores = []
    oobs = []
    for i in list(range(4)):  # average of four
        print("Validation n=50")
        clf = RandomForestClassifier(n_estimators=50, oob_score=True)
        score, imp, oob = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("n=50 fit: top ten important columns:\n", impcol[:10])
        scores.append(score)
        oobs.append(oob)

    print("Valid scores mean, std", np.mean(scores), np.std(scores))
    print("Valid oobs mean, std", np.mean(oobs), np.std(oobs))

    # for test, use optimum validation params, get stats on it
    scores = []
    oobs = []
    for i in list(range(4)):  # average of four
        print("Test n=50")
        clf = RandomForestClassifier(n_estimators=50, oob_score=True)
        score, imp, oob = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("n=50 fit: top ten important columns:\n", impcol[:10])
        scores.append(score)
        oobs.append(oob)

    print("Test scores mean, std", np.mean(scores), np.std(scores))
    print("Test oobs mean, std", np.mean(oobs), np.std(oobs))


