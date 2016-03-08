# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as prfs
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
#    sum(imp) == 1.0
    
    # clfit.fit_transform( X, y=None )  # returns X_new
    
    new_y = clfit.predict( dftest )  # returns predicted Y
    
    test_score = clfit.score( dftest, dftest_y['Y'] )
    print("test score:", test_score, clfit.oob_score_)    
    
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
    
    # how does it know true/false positives, true/false negatives?
    # priors all equal, Y's 1/6?  maybe from confusion matrix?
    # accuracy: percent labeled correctly
    # precision: true positives / (true positives + true negatives)
    # recall:    true positives / (true positives + false negatives)
    precision, recall, fbeta, support = prfs(dftest_y['Y'], new_y)
    print("precision", precision, "\nrecall", recall, \
        "\nfbeta", fbeta, "\nsupport", support)
    
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
    
#    print("Test fit")
#    clf = RandomForestClassifier(n_estimators=100)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("Test fit: Top ten important columns:\n", impcol[:10])

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
    
    print("Validation n=100")
    # vary n_estimators int, oob_score=True,False (n_features=478)
    #   max_features=None,auto,sqrt,log2 # compare w/ n_features
    clf = RandomForestClassifier(n_estimators=100)  # 100 or 500
    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
    impcol = getImportantColumns(dftrain.columns, imp)
    print("n=100 fit: top ten important columns:\n", impcol[:10])
    plotImportances(impcol, plotdir, "v100")
    # sccore 0.9068 0.9099 0.9094
    
#    print("Validation n=100 auto")  # auto is sqrt for classifier
#    clf = RandomForestClassifier(n_estimators=100, max_features='auto')
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 auto fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100auto")
#    # score 0.9051 0.9112 0.9077 0.9079
#    
#    print("Validation n=100 sqrt")  # sqrt(478) = 21.9
#    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 sqrt fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100sqrt")
#    # score 0.9081 0.9051 0.9143 0.9147
#    
#    print("Validation n=100 log2")  # log2(478) = 8.9
#    clf = RandomForestClassifier(n_estimators=100, max_features='log2')
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 log2 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100log")  
#    # score 0.9265 0.9300 0.9283 0.9278
#    
#    print("Validation n=100 9")
#    clf = RandomForestClassifier(n_estimators=100, max_features=9)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 log2 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100_9")  
#    # score 0.9274 0.9287 0.9309
#    
#    print("Validation n=100 22")
#    clf = RandomForestClassifier(n_estimators=100, max_features=22)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 22 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100_22")  
#    # score 0.9121 0.9059 0.9068
#    
#    print("Validation n=100 44")
#    clf = RandomForestClassifier(n_estimators=100, max_features=44)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 44 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100_44")
#    # score 0.9042 0.9029 0.9051
#    
#    print("Validation n=100 100")
#    clf = RandomForestClassifier(n_estimators=100, max_features=100)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 100xx fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100_xx")
#    # score 0.8941 0.8976
#    
#    print("Validation n=100 15")
#    clf = RandomForestClassifier(n_estimators=100, max_features=15)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 15 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100_15")
#    # score 0.9164 0.9134
#    
#    print("Validation n=100 6")
#    clf = RandomForestClassifier(n_estimators=100, max_features=6)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=100 6 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "v100_6")
#    # score 0.9326 0.9357
#    
#    print("Validation n=50")
#    clf = RandomForestClassifier(n_estimators=50)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=50 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "vb0050")  # score 0.9073
#    
#    print("Validation n=200")
#    clf = RandomForestClassifier(n_estimators=200)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=200 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "vb0200")  # score 0.9068
#    
#    print("Validation n=500")
#    clf = RandomForestClassifier(n_estimators=500)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=500 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "vb500")  # score 0.9064
    
#    print("Validation n=1000")
#    clf = RandomForestClassifier(n_estimators=1000)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=1000 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "vb1000")  # score 0.9094
#    
#    print("Validation n=2000")
#    clf = RandomForestClassifier(n_estimators=2000)
#    score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
#    impcol = getImportantColumns(dftrain.columns, imp)
#    print("n=2000 fit: top ten important columns:\n", impcol[:10])
#    plotImportances(impcol, plotdir, "vb2000")  # score 0.9077
    
    for i in list(range(4)):  # average of four
        print("Validation n=100")
        clf = RandomForestClassifier(n_estimators=100, oob_score=True)
        score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("n=100 fit: top ten important columns:\n", impcol[:10])
        # score 0.9160 0.9143 0.9086 0.9077 => 0.912 +- 0.004
        # score 0.9086 0.9073 0.9103 0.9077 => 0.908 +- 0.001
        # np.mean, np.std
        
        print("Validation n=200")
        clf = RandomForestClassifier(n_estimators=200, oob_score=True)
        score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("n=200 fit: top ten important columns:\n", impcol[:10])
        # score 0.9077 0.9073 0.9099 0.9094 => 0.909 +- 0.001
        # score 0.9068 0.9059 0.9112 0.9129 => 0.909 +- 0.003
        
        print("Validation n=500")
        clf = RandomForestClassifier(n_estimators=500, oob_score=True)
        score, imp = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("n=500 fit: top ten important columns:\n", impcol[:10])
        # score 0.9112 0.9094 0.9086 0.9064 => 0.909 +- 0.002
        # score 0.9077 0.9077 0.9081 0.9116 => 0.909 += 0.002
    
    # rather than limited number of columns, optimize other params
    # in validation, then test chosen params with test
    print("Test model fit")
    for i in list(range(4)):  # average of four
        clf = RandomForestClassifier(n_estimators=100, oob_score=True)
        score, imp = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
        print("activity labels:", readActivityLabels())
        impcol = getImportantColumns(dftrain.columns, imp)
        print("Test model fit: top ten important columns:\n", impcol[:10])
        # score 0.9097 0.9155 0.9104 0.9057 => 0.910 +- 0.003
        # score 0.9125 0.9162 0.9131 0.9125 => 0.914 +- 0.002

# test model, for each class:
# precision [ 0.82322357  0.88470067  0.96100279  0.88932039  0.93503937  1. ] 
# recall [ 0.95766129  0.84713376  0.82142857  0.93279022  0.89285714  1. ]
# precision [ 0.83478261  0.88546256  0.96368715  0.89264414  0.91923077  1. ] 
# recall [ 0.96774194  0.85350318  0.82142857  0.91446029  0.89849624  1. ] 
# precision [ 0.82817869  0.89309577  0.95786517  0.90618762  0.92911877  1. ] 
# recall [ 0.97177419  0.85138004  0.81190476  0.92464358  0.91165414  1.  ] 
# precision [ 0.81313131  0.91013825  0.95543175  0.89820359  0.92145594  1. ] 
# recall [ 0.97379032  0.83864119  0.81666667  0.91649695  0.90413534  1. ] 
    
# some thoughts: train, validate, test
# validate excluded from train, used to select between different models
# once a model is selected, use test to test
# get oob from fit, must set beforehand?


