# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import os

from read_clean_data import readRawColumns, readRawData
from clean_predict_allvar import renameColumns


def getPlotDir():
    sns.set_style("darkgrid")
    plotdir = "human_activity_pca_plots/"
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def plot_fit_transform(pca, X, comps, plotname, keys, check_fit=False):
    "do pca fit transform of original data, check components"
    pfit = pca.fit_transform(X)   # class ndarray
    print('  pfit shape', pfit.shape)
    print(pfit[:3])
    
    if check_fit:
        # check components
        print("  PCA comp norm?", np.allclose( \
            list(map(lambda e: sum(e*e), comps)), \
            np.ones(shape=(len(keys)))))
        print("  Orig comp norm?", np.allclose( \
            list(map(lambda e: sum(e*e), comps.T)), \
            np.ones(shape=(len(keys)))))
        
        print("  fit is near equal to dot product?", np.allclose(pfit, np.dot(X, comps.T)))
    #   X columns are orig: [x0, x1, x2]
    #   pfit columns are new: [PCA-0, PCA-1, PCA-2]
    #   so comps.T columns new, rows orig: translates between the two
    #   comps columns orig Xs, rows new PCAs

    # plot transformed data
    plt.clf()
    plt.plot(pfit[:,0], pfit[:,1], 'o', color='blue', alpha=0.3)
    plt.xlabel('PCA-0')
    plt.ylabel('PCA-1')
    plt.title('Human Activity Data by Smartphone, PCA Components')
    plotname += '_fit' + '.png'
    plt.savefig(plotname)

def do_pca_fit(loansData, plotname, keys, rescale=True):
    "do pca fit and fit transform"

    mp = map( lambda k: np.matrix(loansData[k]).T, keys )
    X = np.column_stack(mp)

#    if (rescale):
#        X = StandardScaler().fit_transform(X)

    pca = PCA()
    pout = pca.fit(X)
    
    plot_fit_transform(pca, X, pout.components_, plotname, keys, check_fit=True)
    
    return pca, pout

def plot_comps(pout, plotname, keys):
    "plot pca components in PCA-0, PCA-1 plane"
    comps = pout.components_    # ndarray
    print('  comps shape', comps.shape)
    print(comps)    # print comps[0,:] # print comps[1,:]

    plt.clf()
    compx = comps[0,:]
    compy = comps[1,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(compx, compy, 'o', color='blue', alpha=0.5)
    plt.plot([0.0], [0.0], '+', color='black', alpha=1.0)  # center position
    for i, txt in enumerate(keys):
        ax.annotate(txt, (compx[i], compy[i]), size='x-small')
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2,1.2])
    plt.xlabel('PCA-0')
    plt.ylabel('PCA-1')
    plt.title('Human Activity Data by Smartphone, PCA Components')
    plotname += '_comps' + '.png'
    plt.savefig(plotname)

def plot_var_ratio(pout, plotname, numkeys):
    "plot pca explained variance ratios"
#    varratio = pout.explained_variance_ratio_    # ndarray
    varratio = pout.explained_variance_ratio_[:numkeys]
    varsum = reduce(lambda x,y: x+y, varratio)
    print('  explained_variance_ratio:', varratio, ': sum =', varsum)
    vartotal = (100 * pd.Series(varratio).cumsum()).values
    vartotal = list(filter(lambda x: x<96.0, vartotal))  # cutoff 96% for label
    vartotal = list(map(lambda x: "{:.0f}%".format(x), vartotal))  # python 3 preferred
    print('  vartotal', vartotal)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hscale = len(varratio)
    plt.bar(range(hscale), varratio, color='blue', align='center')
    if numkeys < 40:       # otherwise text too crowded to see
        plt.text(0.7, 0.85, 'Cumulative percentage of    \nexplained variance is shown',
            bbox=dict(edgecolor='black', fill=False), 
            transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
        for i, txt in enumerate(vartotal):
            ax.annotate(txt, (i-0.2+0.2/hscale, varratio[i]+0.002), size='x-small')
    plt.xlim([-0.6, hscale-0.4])
    plt.xlabel('PCA Component Number')
    plt.ylabel('Ratio')
    plt.title('Explained Variance Ratio by Component')
    plotname += '_var' + '.png'
    plt.savefig(plotname)

def do_pca(dftrain, filename, rescale=True):
    "do pca analysis and plots for set of independent variables (keys)"    

    keys = dftrain.columns
    print('do_pca', filename, ': keylen', len(keys))
    plotname = getPlotDir() + filename
    pca, pout = do_pca_fit(dftrain, plotname, keys, rescale)
    plot_comps(pout, plotname, keys)
    plot_var_ratio(pout, plotname, len(keys))
    plot_var_ratio(pout, plotname+"_100", 100)
    plot_var_ratio(pout, plotname+"_030", 30)
    plot_var_ratio(pout, plotname+"_020", 20)
    plot_var_ratio(pout, plotname+"_010", 10)
    
    print('  done: %s' % filename)
    return pca

def simple_pca(dftrain, dftest, ncomps=30):
    pca = PCA(n_components=ncomps)
    pca.fit(dftrain)
    X_train = pca.transform(dftrain)
    X_test = pca.transform(dftest)
#    print("pca X_shapes", X_train.shape, X_test.shape)
    return X_train, X_test

def do_svm(dftrain, dftrain_y, dftest, dftest_y, label=''):
    clf = LinearSVC()
    print("fit shapes", dftrain.shape, dftrain_y.shape, dftest.shape, dftest_y.shape)
    clf.fit(dftrain, dftrain_y['Y'])
    fit_score = clf.score(dftrain, dftrain_y['Y'])
    p_score = clf.score(dftest, dftest_y['Y'])
#    print("fit score", fit_score)
    pred_y = clf.predict( dftest )
    pred_correct = sum(pred_y == dftest_y['Y'])
    pred_score = pred_correct/dftest_y.shape[0]
    print("%s: svm fit score %.5f, predict score %.5f %.5f" % (label, fit_score, pred_score, p_score))


def main():
    dfcol, dups = readRawColumns()
    dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
    
    dftrain = renameColumns(dftrain)
    dftest = renameColumns(dftest)
    print("dftrain shape head", dftrain.shape, "\n", dftrain[:5])
    print("dftest shape head", dftest.shape, "\n", dftest[:5])
    
    pca = do_pca(dftrain, "all", rescale=False)

    do_svm(dftrain, dftrain_y, dftest, dftest_y, 'raw data, all cols')
    do_svm(dftrain.ix[:, :30], dftrain_y, dftest.ix[:, :30], dftest_y, 'raw data, 30 cols')
    
#    X_train, X_test = simple_pca(dftrain, dftest, ncomps=30)
#    do_svm(X_train, dftrain_y, X_test, dftest_y)
#    X_train, X_test = simple_pca(dftrain, dftest, ncomps=50)
#    do_svm(X_train, dftrain_y, X_test, dftest_y)
#    X_train, X_test = simple_pca(dftrain, dftest, ncomps=100)
#    do_svm(X_train, dftrain_y, X_test, dftest_y)
    
    X_train, X_test = simple_pca(dftrain, dftest, ncomps=562)
    do_svm(X_train[:, :30], dftrain_y, X_test[:, :30], dftest_y, 'pca 30 cols')
    do_svm(X_train[:, :50], dftrain_y, X_test[:, :50], dftest_y, 'pca 50 cols')
    do_svm(X_train[:, :100], dftrain_y, X_test[:, :100], dftest_y, 'pca 100 cols')


if __name__ == '__main__':
    main()

