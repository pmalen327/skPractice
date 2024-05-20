import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings 
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV


# 1 = heart disease
# 0 = healthy


df = pd.read_csv('data.csv')
categories = list(df.columns)
X = df.drop('target', axis=1)
y = df.iloc[:,-1:].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=69)

with ignore_warnings(category=[ConvergenceWarning, FitFailedWarning]): 

    names = ['svc', 'sgd classifier']
    classifiers = [svm.SVC(), SGDClassifier()]

    svm_params = {
        'C':[.25, .5, 1.0],
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'degree':[2, 3, 4, 5],
        'gamma':['scale', 'auto']
        }

    sgd_params = {
        'loss':['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron',
                'squared_error', 'huber', 'epsilon_intensive', 'squared_epsilon_intensive'],
        'penalty':['l2', 'l1', 'elasticnet'],
        'alpha':[1e-4, 1e-3, 1e-2],
        'max_iter':[1000, 1500, 2000],
        'shuffle':[True, False],
        'epsilon':[1e-2, .1, .2, .5],
        'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive']
        }

    parameters = [svm_params, sgd_params]

    for i in range(len(classifiers)):
        search = HalvingGridSearchCV(classifiers[i], parameters[i], random_state=69).fit(X,y)
        print(f'{names[i]} achieved an accuracy of {round(search.best_score_, 4)} \n with the parameters:{search.best_params_}')