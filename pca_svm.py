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


def get_plotdir():
    plotdir = "human_activity_pca_plots/"
    return plotdir

def make_plotdir():
    sns.set_style("darkgrid")
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def plot_transform(pfit, X, comps, plotname, keys, check_fit=True):
    "do pca fit transform of original data, check components"
    print('  pfit shape', pfit.shape)
    print(pfit[:3])
    
    if check_fit:
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

def plot_pca_svm(preds):
    print("preds", preds)
    px = [e[0] for e in preds]
    py = [e[1] for e in preds]
    hscale = len(py)
    pylabel = list(map(lambda y: "{:.3f}".format(y), py))
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    plt.bar(px, py, width=8, tick_label=px, color='blue', align='center')
    plt.bar(range(len(py)), py, tick_label=px, color='blue', align='center')
    for i, txt in enumerate(pylabel):
        ax.annotate(txt, (i-0.2+0.2/hscale, py[i]+0.01), size='small')
        
    plt.ylim([0.5, 1.0])
    plt.xlabel('Number of PCA Components')
    plt.ylabel('SVM Predict Score')
    plt.title('Human Activity by Smartphone Data, SVM with PCA')
    
    plotname = get_plotdir() + 'pca_svm.png'
    plt.savefig(plotname)

def plot_comps(pout, plotname, keys):
    "plot pca components in PCA-0, PCA-1 plane"
    comps = pout.components_    # ndarray
    print('  comps shape', comps.shape)
    print(comps)    # print comps[0,:] # print comps[1,:]

    plt.clf()
    compx = comps[0,:]
    compy = comps[1,:]
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
    plt.plot(compx, compy, 'o', color='blue', alpha=0.5)
    plt.plot([0.0], [0.0], '+', color='black', alpha=1.0)  # center position
#    for i, txt in enumerate(keys):
#        ax.annotate(txt, (compx[i], compy[i]), size='x-small')
    plt.xlim([-0.02,0.02])
    plt.ylim([-0.2,0.2])
    plt.xlabel('PCA-0')
    plt.ylabel('PCA-1')
    plt.title('Human Activity by Smartphone Data, PCA Components')
    plotname += '_comps' + '.png'
    plt.savefig(plotname)

def plot_var_ratio(pout, plotname, numkeys):
    "plot pca explained variance ratios"
#    varratio = pout.explained_variance_ratio_    # ndarray
    varratio = pout.explained_variance_ratio_[:numkeys]
    varsum = reduce(lambda x,y: x+y, varratio)
    print('  explained_variance_ratio:', varratio[:3], '...', varratio[-3:], ': sum =', varsum)
    vartotal = (100 * pd.Series(varratio).cumsum()).values
    vartotal = list(filter(lambda x: x<96.0, vartotal))  # cutoff 96% for label
    vartotal = list(map(lambda x: "{:.0f}%".format(x), vartotal))  # python 3 preferred
    print('  vartotal', vartotal)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hscale = len(varratio)
    plt.bar(range(hscale), varratio, color='blue', align='center')
    plt.text(0.7, 0.94, 'Cumulative sum of explained variance: {:.1f}%'.format(varsum*100),
        bbox=dict(edgecolor='black', fill=False), 
        transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
    if numkeys < 40:       # otherwise text too crowded to see
        plt.text(0.8, 0.25, 'Cumulative percentage of    \nexplained variance is shown',
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

def plot_comps_vars(pout, plotname, keys):
    plot_comps(pout, plotname, keys)
    plot_var_ratio(pout, plotname, len(keys))
    plot_var_ratio(pout, plotname+"_100", 100)
    plot_var_ratio(pout, plotname+"_030", 30)
    plot_var_ratio(pout, plotname+"_020", 20)
    plot_var_ratio(pout, plotname+"_010", 10)

def explore_pca(dftrain, dftest, filename):
    "do pca analysis and plots for set of independent variables (keys)"    

    keys = dftrain.columns
    print('do_pca', filename, ': keylen', len(keys))
    plotname = get_plotdir() + filename
    
    pca = PCA()
    pout = pca.fit(dftrain)
    X_train = pca.transform(dftrain)
    X_test = pca.transform(dftest)
    
    plot_transform(X_train, dftrain, pout.components_, plotname, keys)
    plot_comps_vars(pout, plotname, keys)
    
    return X_train, X_test

def quick_pca(dftrain, dftest, ncomps=100):
    pca = PCA(n_components=ncomps)
    pca.fit(dftrain)
    X_train = pca.transform(dftrain)
    X_test = pca.transform(dftest)
    return X_train, X_test

def do_svm(dftrain, dftrain_y, dftest, dftest_y, label=''):
    clf = LinearSVC()
    print("fit shapes", dftrain.shape, dftrain_y.shape, dftest.shape, dftest_y.shape)
    clf.fit(dftrain, dftrain_y['Y'])
    fit_score = clf.score(dftrain, dftrain_y['Y'])
    pred_score = clf.score(dftest, dftest_y['Y'])
    print("%s: svm fit score %.5f, predict score %.5f" % (label, fit_score, pred_score))
    return pred_score


def main():
    dfcol, dups = readRawColumns()
    dftrain, dftrain_y, dftest, dftest_y = readRawData(dfcol)
    
    dftrain = renameColumns(dftrain)
    dftest = renameColumns(dftest)
    print("dftrain shape head", dftrain.shape, "\n", dftrain[:3])
    print("dftest shape head", dftest.shape, "\n", dftest[:3])
    
    make_plotdir()
    explore_pca(dftrain, dftest, "all")    # 562 columns

    do_svm(dftrain, dftrain_y, dftest, dftest_y, 'raw data, all cols')
    do_svm(dftrain.ix[:, :30], dftrain_y, dftest.ix[:, :30], dftest_y, 'raw data, 30 cols')
    
    X_train, X_test = quick_pca(dftrain, dftest, ncomps=100)
    
    preds = []
    for j in [10, 20, 30, 50, 100]:
        p = do_svm(X_train[:, :j], dftrain_y, X_test[:, :j], dftest_y, 'pca {:.0f} cols'.format(j))
        preds.append((j, p))
   
    plot_pca_svm(preds)


if __name__ == '__main__':
    main()

# to do:  cross-validation, confusion matrix

