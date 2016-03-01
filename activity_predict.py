# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
#    pat5 = re.compile('(Body|Mag|,)')  # 469
    pat5 = re.compile('(Body|,)')    # 477
    
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
    tt = re.sub(pat5, '', tt)  # removal chnages dup count to 477
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

def check_duplicate_columns(dups, dfcol):
    '''check duplicate columns'''
    print("DUPS")
    dc = dfcol['label2']
    for dup in dups:
        dg = list(filter(lambda s: s.startswith(dup), dc))
        dt = dftrain[dg]
        print("dt dup %s mean" % dup)
        print(dt.mean())
# values, mean are close but not identical, within 3-4 places

def readRawColumns():
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

    print('dfcol\n', dfcol[:5])    
    print('dfcol shape', dfcol.shape)
    clist = list(dfcol['label'])
    
    print('head clist', len(clist), clist[:5])  # 561
    cset = set(clist)
    print('head cset', len(cset), list(cset)[:5])  # 469
    print('done')

    # check duplicate columns    
    dups = [k for (k,v) in hh.items() if v > 1]
    dups = sorted(dups)
    check_duplicate_columns(dups, dfcol)
    
    return dfcol

def readActivityLabels():
    '''read activity labels'''
    dfact = pd.read_table(datadir + "activity_labels.txt", \
        sep='\s+', header=None, index_col=0)
    dfact.columns=['act']
    print("dfact\n", dfact)
    return dfact

def readRawTrainData(dfcol):
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
    print("dftrain head\n", dftrain[:5])
    print("dftrain_y shape", dftrain_y.shape, "head\n", dftrain_y[295:305])
    return dftrain, dftrain_y

def readRawTestData(dfcol):
    testdir = datadir + "test/"
    dftest = pd.read_table(testdir + "X_test.txt", \
        sep='\s+', header=None, names=dfcol['label2'])
    dftest['subject'] = pd.read_table(testdir + "subject_test.txt", \
        sep='\s+', header=None)
    dftest_y = pd.read_table(testdir + "y_test.txt", \
        sep='\s+', header=None, names=['Y'])
    dftest_y['activity'] = dftest_y['Y'].apply(lambda x: \
        dfact['act'][dfact.index[x-1]])
    print("dftest head", dftest.shape, "\n", dftest[:5])
    print("dftest_y shape", dftest_y.shape, "head\n", dftest_y[:5])
    return dftest, dftest_y

def rfFitScore(clf, dftrain, dftest):
    '''random forest classifier fit and score.
       clf=RandomForestClassifier, dftrain=train data,
       dftest=test data'''
    
    clfit = clf.fit(dftrain, dftrain_y['Y'])  # clf.fit(X, y)
    
    imp = clfit.feature_importances_  # ndarray of 562
    print("importances", imp.shape, "\n", imp[:12], "\n...\n", imp[-12:])
    print("sorted imps\n", (sorted(imp))[-20:])   
    
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
    print("test predict probabilities head:\n", new_p[:12])
    
    # cross table of variable predictions
    ptab = pd.crosstab(dftest_y['Y'], new_y, \
        rownames=['actual'], colnames=['predicted'])
    print("cross table:\n", ptab)
    return test_score, imp

dfcol = readRawColumns()
dfact = readActivityLabels()
dftrain, dftrain_y = readRawTrainData(dfcol)
dftest, dftest_y = readRawTestData(dfcol)

dftrain['tAccMag_Mean'].hist(by=dftrain_y['activity'])
dftrain['fAccMag_Mean'].hist(by=dftrain_y['activity'])
dftrain['tGyroMag_Mean'].hist(by=dftrain_y['activity'])
dftrain['fBGyroMag_Mean'].hist(by=dftrain_y['activity'])

# try different random forest parameters
# try different data cleaning versions
clf = RandomForestClassifier(n_estimators=10)
score, imp = rfFitScore(clf, dftrain, dftest)

def getImportantColumns(dfcol, imp, level=0.01):
    '''sort column names by RandomForest importance
       for use in dftrain, dftest subset'''
    cslist = sorted(zip(imp, list(dfcol['label2'])), reverse=True)
    cilist = list(filter(lambda e: e[0] > level, cslist))
    return list(map(lambda e: e[1], cilist))

impcol = getImportantColumns(dfcol, imp, 0.01)

# score 0.901
# Cross table shows ~10 percent covariance within
#   sedentary activities (LAYING SITTING STANDING)
#   and within active activities (WALKING UPSTAIRS DOWNSTAIRS),
#   but almost no covariance between active and 
#   sedentary activities.

# clf = RandomForestClassifier(n_estimators=20)
# rfFitScore(clf, dftrain, dftest)
# score 0.913

# clf = RandomForestClassifier(n_estimators=5)
# rfFitScore(clf, dftrain, dftest)
# score 0.896

# score almost unchanged, maybe 0.5 to 1.0 %


