# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:07:15 2022

@author: Fabrizio
"""
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score,roc_auc_score,precision_recall_fscore_support,auc
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV, cross_val_score,KFold,learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS

from sklearn import preprocessing
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model






def plot_roc_curve(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
          

        
def fit_ml_algo(algo, x_train, y_train, x_val, cv):
    
    model = algo.fit(x_train, y_train)
    test_pred = model.predict(x_val)
    if (isinstance(algo, (LogisticRegression, 
                          KNeighborsClassifier, 
                          GaussianNB, 
                          MultinomialNB,
                          DecisionTreeClassifier, 
                          RandomForestClassifier,
                          SVC,
                          #eclf
                          ))):
        probs = model.predict_proba(x_val)[:,1]
    else:
        probs = "Not Available"
    acc = round(model.score(x_val, y_val) * 100, 2) 
    # CV 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  x_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, test_pred, acc, acc_cv, probs


def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def l_c (clf,x_train,y_train)  :        
     train_sizes, train_scores, valid_scores = learning_curve(clf
        , x_train, y_train, train_sizes=[300,500,600,700,800,900,1000,1500,2000,2500,3000,3200], cv=5,scoring='roc_auc')
    
    
     t_scores=[]
    
     for scores in train_scores:
        t_scores.append(np.mean(scores))
    
     v_scores=[]
    
     for scores in valid_scores:
        v_scores.append(np.mean(scores))
    
     x=train_sizes
    
     fig, ax = plt.subplots(1, figsize=(6, 3))
    
    
     fig.suptitle(clf, fontsize=15)
     ax.plot(x, t_scores,"o-", color="red", label="Training Scores")
     ax.plot(x, v_scores,"o-", color="g", label="Validation Scores")
     ax.grid(True)
     ax.legend(loc="center left", title="Legend", frameon=False)
     plt.show()