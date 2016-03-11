# -*- coding: utf-8 -*-

# predict human activity after reading and cleaning data
# use Random Forest w/ all anonymous variable names

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from read_clean_data import readRawColumns, readRawData, rfFitScore, \
    getImportantColumns, getPlotDir

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

def plot_confusion_matrix(cm, plotdir, label, target_names):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title='Random Forest: Confusion matrix ' + label
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)  # rotation=45
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()  # adds padding
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(plotdir + label + '_conf_mat')

def gridscore_boxplot(gslist, plotdir, label, xlabel):
    vals = list(map(lambda e: e.cv_validation_scores, gslist))
    labs = list(map(lambda e: list(e.parameters.values()), gslist))
    labs = list(map(lambda e: reduce(lambda a,b: str(a)+"\n"+str(b), e), labs))
    xpar = list(gslist[0].parameters.keys())
    xpar = reduce(lambda a,b: a+", "+b, xpar)
    plt.clf()
    plt.boxplot(vals, labels=labs)
    plt.title("Human Activity Predicted by Random Forest")
    plt.xlabel(xpar + " (with " + xlabel + ")")
    plt.ylabel("Fraction Correct")
    plt.savefig(plotdir + "gridscore_" + label)

if __name__ == '__main__':
    dfcol, dups = readRawColumns()
    dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
    
    dftrain = rename_columns(dftrain)
    dftest = rename_columns(dftest)
    print("dftrain shape head", dftrain.shape, "\n", dftrain[:5])
    print("dftest shape head", dftest.shape, "\n", dftest[:5])
    
    plotdir = getPlotDir()
    
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
    for i in list(range(5)):  # average of five
        print("Validation n=50")
        clf = RandomForestClassifier(n_estimators=50, oob_score=True)
        score, imp, oob = rfFitScore(clf, dftrain, dftrain_y, dfvalid, dfvalid_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("n=50 fit: top ten important columns:\n", impcol[:10])
        scores.append(score)
        oobs.append(oob)

    print("Valid scores mean, std", np.mean(scores), np.std(scores))
    print("Valid oobs mean, std", np.mean(oobs), np.std(oobs))
    
    clf = RandomForestClassifier(n_estimators=50, oob_score=True)
    # don't need to split train, valid
    scores = cross_validation.cross_val_score(clf, \
        dftrain, dftrain_y['Y'], cv=4)
    print("\nCV scores:", scores)
    print("CV scores mean, std", np.mean(scores), np.std(scores))
    
    # find optimum parameters if possible
    clf = RandomForestClassifier(oob_score=True)  # n_estimators=50
    param_grid = [{'n_estimators': [50, 100, 200], \
      'max_features': [None, 'sqrt', 'log2'] }]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, \
      verbose=1, n_jobs=-1)    # verbose=10
    gs.fit(dftrain, dftrain_y['Y'])
    new_y = gs.predict(dftest)
    print("gs score %.4f (%d of %d)" % (gs.score(dftest, dftest_y['Y']), \
      sum(new_y == dftest_y['Y']), dftest_y.shape[0] ))
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.4f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
#    print("gs params", gs.get_params())
    # all mean within 2 * std of each other, best not meaningful 
    gridscore_boxplot(gs.grid_scores_, plotdir, \
        "multi_opt", "oob_score=True")
    print("\nclassification_report\n", classification_report(dftest_y['Y'], new_y))
    cm = confusion_matrix(dftest_y['Y'], new_y)
    plot_confusion_matrix(cm, plotdir, 'opt', list(sorted(set(dftest_y['Y']))))
    
    # for test, use optimum validation params, get stats on it
    scores = []
    oobs = []
    for i in list(range(5)):  # average of five
        print("Test optimum")
        clf = RandomForestClassifier(n_estimators=50, oob_score=True)
#        clf = gs.best_estimator_
        score, imp, oob = rfFitScore(clf, dftrain, dftrain_y, dftest, dftest_y)
        impcol = getImportantColumns(dftrain.columns, imp)
        print("opt fit: top ten important columns:\n", impcol[:10])
        scores.append(score)
        oobs.append(oob)

    print("Test scores mean, std", np.mean(scores), np.std(scores))
    print("Test oobs mean, std", np.mean(oobs), np.std(oobs))

# end output copied

# find original labels of importances
#  could do for each in opt list, take set intersection of top 10
#  but almost no difference in opt parameters, just get a sense of it
    impcol2 = getImportantColumns(dfcol['label2'], imp)
    print("Top ten importances of last opt fit, by original name:\n", \
        list(map(lambda e: e[1], impcol2))[:10] )

