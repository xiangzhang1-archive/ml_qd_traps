"""
Trap classification with support vector machine.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
from sklearn import svm
from sklearn import preprocessing

__all__ = ['TrapClassify']

class ClassifySVM:

    def __init__(self, data_feats, target_feats, data_feat_labels, target_feat_labels):
        self.data_feats = data_feats
        self.target_feats = target_feats
        self.data_feat_labels = data_feat_labels
        self.target_feat_labels = target_feat_labels

##-------------------------------------------------------------------------------------------------
    def svm_2d_plot(self, sel_data_feat_labels, sel_target_feat_labels, clf_params, plt_params):

        sel_data_feats = []
        for fid in range(len(sel_data_feat_labels)):
            sel_data_feats.append(self.data_feats[self.data_feat_labels.index(sel_data_feat_labels[fid])])
        nrecord = len(sel_data_feats[0])
        nfeat = len(sel_data_feats)
        X = np.array([[sel_data_feats[j][i] for j in range(nfeat)] for i in range(nrecord)])
        y = np.array(self.target_feats[self.target_feat_labels.index(sel_target_feat_labels[0])])
            
        C = clf_params['C']
        kernel = clf_params['kernel']
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clf = svm.SVC(kernel=kernel, C=C).fit(X_scaled, y)
        
        margin_scaled = plt_params['margin_scaled']
        npt = plt_params['npt']
        x_min_scaled, x_max_scaled = X_scaled[:, 0].min()-margin_scaled, X_scaled[:, 0].max()+margin_scaled
        y_min_scaled, y_max_scaled = X_scaled[:, 1].min()-margin_scaled, X_scaled[:, 1].max()+margin_scaled
        x_margin = margin_scaled * (X[:, 0].max() - X[:, 0].min()) / (X_scaled[:, 0].max() - X_scaled[:, 0].min())
        y_margin = margin_scaled * (X[:, 1].max() - X[:, 1].min()) / (X_scaled[:, 1].max() - X_scaled[:, 1].min())
        x_min, x_max = X[:, 0].min()-x_margin, X[:, 0].max()+x_margin
        y_min, y_max = X[:, 1].min()-y_margin, X[:, 1].max()+y_margin
        x_h_scaled = (x_max_scaled - x_min_scaled) / npt
        y_h_scaled = (y_max_scaled - y_min_scaled) / npt
        x_h = (x_max - x_min) / npt
        y_h = (y_max - y_min) / npt
        xx_scaled, yy_scaled = np.meshgrid(np.arange(x_min_scaled, x_max_scaled, x_h_scaled), \
                                           np.arange(y_min_scaled, y_max_scaled, y_h_scaled))
        xx, yy = np.meshgrid(np.arange(x_min, x_max, x_h), np.arange(y_min, y_max, y_h))
        title = sel_target_feat_labels[0] + ', SVC w. ' + kernel + ' kernel'
        fig = plt.figure()
        Z = clf.predict(np.c_[xx_scaled.ravel(), yy_scaled.ravel()])
        Z = Z.reshape(xx_scaled.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel(sel_data_feat_labels[0], size=plt_params['label_size'])
        plt.ylabel(sel_data_feat_labels[1], size=plt_params['label_size'])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(size=plt_params['tick_size'])
        plt.yticks(size=plt_params['tick_size'])
        plt.title(title, size=plt_params['label_size'])
        plt.axis('tight')
        plt.show()

        
        
        
                                                                                
        
