# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from functools import reduce
import os
import re

datadir = "data/UCI_HAR_Dataset/"

# should be a way to do this all in one regex
def parseColumn(ss):
    pat1 = re.compile('[\(\)]')
    pat2 = re.compile('(.*)(\d+),(\d+)')
    pat3 = re.compile('(.*)([XYZ]),([XYZ1234])')
    pat4 = re.compile('[-]')
    pat5 = re.compile('(Body|Mag|,)')  # 469
#    pat5 = re.compile('(Body|,)')    # 477
    
    def splitJoin(pat, s):
        t = re.split(pat, s)
        return reduce(lambda a,b: a+b, t)
    
    def splitJoinNum(s):
        '''join two numbers with underscore'''
        t = re.split(pat2, s)
        if len(t) > 4:
            u = t[0] + t[1] + t[2] + '_' + t[3]
        else:
            u = reduce(lambda a,b: a+b, t)
        return u
    
    tt = re.sub(pat1, '', ss)
    tt = splitJoinNum(tt)
    tt = splitJoin(pat3, tt)
    tt = re.sub(pat4, '_', tt)
    tt = tt.replace('BodyBody', 'BBody')
    tt = re.sub(pat5, '', tt)  # removal changes dup count to 477
    tt = tt.replace('mean','Mean')
    tt = tt.replace('std','Std')
    tt = tt.replace('gravity','_gravity')
    tt = tt.replace('angle','angle_')
    return tt

# test parseColumn()
# ss = 'tBodyGyroJerk(Mag)-arCoeff()4)'
# ss = 'fBodyAcc-mad()-Y'
# ss = 'tBodyAccJerk-correlation()-Y,Z'
# ss = 'tBodyGyroJerk-arCoeff()-X,3'
# ss = 'fBodyBodyGyroMag-max()'
# ss = 'fBodyAcc-bandsEnergy()-9,16'
# ss = 'angle(tBodyAccJerkMean),gravityMean)'
# ss = 'fBodyAcc-max()-Z'
# ss = 'fBodyAcc-mean()-X'
# ss = 'tGravityAccMag-std()'

# tt = parseColumn(ss)
# print('tt', tt)

def readRawColumns(printOut=False):
    '''read raw data columns'''
    feature_file = datadir + "features.txt"
    dfcol = pd.read_csv(feature_file, sep='\s+', header=None, index_col=0)
    dfcol.columns=['label']
    
    dfcol['label'] = dfcol['label'].apply(lambda s: parseColumn(s))
    
    # make unique column names, assign to label2
    hh = {}
    dlist = []
    for c in dfcol['label']:
        if c in hh.keys():
            hh[c] += 1
            dlist.append(c + '_' + str(hh[c]))
        else:
            hh[c] = 0
            dlist.append(c)
    dfcol['label2'] = dlist

    if (printOut==True):
        print('dfcol\n', dfcol[:5])    
        print('dfcol shape', dfcol.shape)

    # check duplicate columns    
    dups = [k for (k,v) in hh.items() if v > 1]
    dups = sorted(dups)
    
    return dfcol, dups

def readActivityLabels():
    '''read activity labels'''
    dfact = pd.read_table(datadir + "activity_labels.txt", \
        sep='\s+', header=None, index_col=0)
    dfact.columns=['act']
    return dfact

def readRawTrainData(dfcol, dfact, printOut=False):
    '''read in raw train data'''
    traindir = datadir + "train/"
    dftrain = pd.read_table(traindir + "X_train.txt", \
        sep='\s+', header=None, names=dfcol['label2'])
    dftrain['subject'] = pd.read_table(traindir + "subject_train.txt", \
        sep='\s+', header=None)
    dftrain_y = pd.read_table(traindir + "y_train.txt", \
        sep='\s+', header=None, names=['Y'])
    dftrain_y['activity'] = dftrain_y['Y'].apply(lambda x: \
        dfact['act'][dfact.index[x-1]])
    if (printOut==True):
        print("dftrain head\n", dftrain[:5])
        print("dftrain_y shape", dftrain_y.shape, "head\n", dftrain_y[295:305])
    return dftrain, dftrain_y

def readRawTestData(dfcol, dfact, printOut=False):
    testdir = datadir + "test/"
    dftest = pd.read_table(testdir + "X_test.txt", \
        sep='\s+', header=None, names=dfcol['label2'])
    dftest['subject'] = pd.read_table(testdir + "subject_test.txt", \
        sep='\s+', header=None)
    dftest_y = pd.read_table(testdir + "y_test.txt", \
        sep='\s+', header=None, names=['Y'])
    dftest_y['activity'] = dftest_y['Y'].apply(lambda x: \
        dfact['act'][dfact.index[x-1]])
    if (printOut==True):
        print("dftest head", dftest.shape, "\n", dftest[:5])
        print("dftest_y shape", dftest_y.shape, "head\n", dftest_y[:5])
    return dftest, dftest_y

def readRawData(dfcol, printOut=False):
    dfact = readActivityLabels()
    dftrain, dftrain_y = readRawTrainData(dfcol, dfact, printOut)
    dftest, dftest_y = readRawTestData(dfcol, dfact, printOut)
    return dftrain, dftrain_y, dftest, dftest_y

def check_duplicate_columns(dfcol, dups):
    '''check duplicate columns'''
    print("DUPS, len =", len(dups))
    for dup in dups:
        dg = list(filter(lambda s: s.startswith(dup), \
            dfcol['label2']))
        dt = dftrain[dg]
        print("dt dup %s mean" % dup)
        print(dt.mean())
# values, mean are close but not identical, within 3-4 places

def removeDuplicateColumns(dfcol, dups, dftrain, dftest):
    '''find duplicate columns to remove from dataframe'''
    dg = list(map(lambda dup: list(filter(lambda s: \
        s.startswith(dup+"_"), dfcol['label2'])), dups))
    dg = [item for sublist in dg for item in sublist]
    dftrain = dftrain.drop(dg, axis=1)
    dftest = dftest.drop(dg, axis=1)
    return dftrain, dftest

def rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y):
    '''random forest classifier fit and score.
       clf=RandomForestClassifier, dftrain=train data,
       dftrain_y=train data Y, dftest=test data,
       dftest_y=test data Y'''
    
    clfit = clf.fit(dftrain, dftrain_y['Y'])  # clf.fit(X, y)
    
    imp = clfit.feature_importances_  # ndarray of 562
#    print("importances", imp.shape, "\n", imp[:12], "\n...\n", imp[-12:])
#    print("sorted imps\n", (sorted(imp))[-20:])
#    print("clfit params", clfit.get_params)
    
    # clfit.fit_transform( X, y=None )  # returns X_new
    
    new_y = clfit.predict( dftest )  # returns predicted Y
    
    test_score = clfit.score( dftest, dftest_y['Y'] )
    print("test score:", test_score)    
    
    # calculate test score by other means
    print("test predict True %.3f percent, %d out of %d" % \
      ((100 * sum(dftest_y['Y'] == new_y) / dftest_y.shape[0]), \
       sum(dftest_y['Y'] == new_y), dftest_y.shape[0]))
    print("test predict False %.3f percent, %d out of %d" % \
      ((100 * sum(dftest_y['Y'] != new_y) / dftest_y.shape[0]), \
       sum(dftest_y['Y'] != new_y), dftest_y.shape[0]))
    
    new_p = clfit.predict_proba( dftest )
    # probability of each X variable to predict each y class
    print("test predict probabilities head:\n", new_p[:5])
    
    # cross table of variable predictions
    ptab = pd.crosstab(dftest_y['Y'], new_y, \
        rownames=['actual'], colnames=['predicted'])
    print("cross table:\n", ptab)
    return test_score, imp

def getImportantColumns(dftraincol, imp):
    '''sort column names by RandomForest importance
       for use in dftrain, dftest subset'''
    return sorted(zip(imp, dftraincol), reverse=True)

def getPlotDir():
    plotdir = "human_activity_plots/"
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def plotImportances(impcol, plotdir, label):
    vals = list(map(lambda e: e[0], impcol))
    valsum = np.cumsum(vals)
    vals = vals / vals[0]     # norm
    
    plt.clf()
    plt.plot(range(len(vals)), vals)
    plt.plot(range(len(valsum)), valsum)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Feature Number")
    plt.ylabel("Importance")
    plt.title("Random Forest Importances")
    plt.legend(('relative importance', 'cumulative importance'), \
        loc='center right')
    plt.savefig(plotdir + "impt_" + label)    
    
    plt.clf()
    plt.plot(range(100), vals[:100])
    plt.plot(range(100), valsum[:100])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Feature Number")
    plt.ylabel("Importance")
    plt.title("Random Forest Importances")
    plt.legend(('relative importance', 'cumulative importance'), \
        loc='upper center')
    plt.text(70, 0.35, "First Hundred\nImportances")
    plt.savefig(plotdir + "impt100_" + label)

def plotHistograms(dftrain, dftrain_y, plotdir):
    labels = ['tAcc_Mean', 'fAcc_Mean', 'tGyro_Mean', 'fBGyro_Mean']
    for label in labels:
        plt.clf()
        dftrain[label].hist(by=dftrain_y['activity'], \
            sharex=True, xlabelsize=8, ylabelsize=8)
        plt.text(-2.5, -170, "Histograms of " + label + " by Activity")
        plt.savefig(plotdir + "hist_" + label)

if __name__ == '__main__':
    dfcol, dups = readRawColumns(printOut=True)
    dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol, printOut=True)
    
    plotdir = getPlotDir()
    plotHistograms(dftrain, dftrain_y, plotdir)
    
    # first analysis: remove columns w/ duplicate names
    print("\nRemove columns with duplicate names")
    dftrain, dftest = removeDuplicateColumns(dfcol, dups, dftrain, dftest)
    print("dftrain head", dftrain.shape, "\n", dftrain[:5])    
    
    # check that random forest works
    print("Basic check")
    clf = RandomForestClassifier(n_estimators=10)
    score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
    impcol = getImportantColumns(dftrain.columns, imp)
    print("Basic check: Top ten important columns:\n", impcol[:10])

# score .903
# Cross table shows ~10 percent covariance within
#   sedentary activities (LAYING SITTING STANDING)
#   and within active activities (WALKING UPSTAIRS DOWNSTAIRS),
#   but almost no covariance between active and 
#   sedentary activities.

#   should give overfitting
    print("Overfit")
    clf = RandomForestClassifier(n_estimators=20)
    score, imp = rfFitScore(clf, dftrain, dftrain_y, dftrain, dftrain_y)
    impcol = getImportantColumns(dftrain.columns, imp)
    print("Overfit: Top ten important columns:\n", impcol[:10])
    
    print("Test fit")
    clf = RandomForestClassifier(n_estimators=100)
    score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
    impcol = getImportantColumns(dftrain.columns, imp)
    print("Test fit: Top ten important columns:\n", impcol[:10])
    
    # how do I know true/false positives, true/false negatives?
    # to calculate precision, recall?  priors all equal, Y's 1/6?
    # accuracy: percent labeled correctly
    # precision: true positives / (true positives + true negatives)
    # recall:    true positives / (true positives + false negatives)

#    split training, test sets into train, validate, test
#    use dftrain, dfvalid to find top ten columns => gives
#       new model, new dftrain then dftest
#    for real tests use 500 estimators

# add subject to Y's for validation split
    cutoff = 23
    dftrain_y['subject'] = dftrain['subject']
    dfvalid = dftrain[dftrain['subject'] > cutoff]
    dftrain = dftrain[dftrain['subject'] <= cutoff]
    dfvalid_y = dftrain_y[dftrain_y['subject'] > cutoff]
    dftrain_y = dftrain_y[dftrain_y['subject'] <= cutoff]
    dftrain_y = dftrain_y.drop(['subject'], axis=1)
    dfvalid_y = dfvalid_y.drop(['subject'], axis=1)
    
#    print("validation: dftrain head", dftrain.shape, "\n", dftrain[:5])
#    print("validation: dfvalid head", dfvalid.shape, "\n", dfvalid[:5])
#    print("validation: dftrain_y head", dftrain_y.shape, "\n", dftrain_y[:5])
#    print("validation: dfvalid_y head", dfvalid_y.shape, "\n", dfvalid_y[:5])
    
    print("Validation fit")
    clf = RandomForestClassifier(n_estimators=100)
    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
    impcol = getImportantColumns(dftrain.columns, imp)
#    sum(list(map(lambda e: e[0], impcol)))  # equals 1.0
    print("Validation fit: Top twenty important columns:\n", impcol[:20])
    # plot importances
    plotImportances(impcol, plotdir, "vfit")
    
    # model after training: dftrain w/ top 10 or 20 importances?
    # top 10 ~ 60-70% accuracy, top 20 ~ 80% accuracy, better cross table
    impcolnames = list(map(lambda e: e[1], impcol[:20]))
    dftrain = dftrain[impcolnames]
    dftest = dftest[impcolnames]
#    print("validation: dftrain model head", dftrain.shape, "\n", \
#        dftrain[:5], "\n", dftest[:5])
    
    print("Test model fit")
    clf = RandomForestClassifier(n_estimators=100)
    score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
    print("activity labels:", readActivityLabels())
    impcol = getImportantColumns(dftrain.columns, imp)
    print("Test model fit: Top ten important columns:\n", impcol[:10])
    
# some thoughts: train, validate, test
# validate excluded from train, used to select between different models
# once a model is selected, use test to test
# priors? check distribution of activities in train, test => 1/6
#     from confusion matrix?
# get oob from fit, must set beforehand?


