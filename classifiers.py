import pandas as pd
import numpy as np
from sklearn import svm
from scipy.stats import wasserstein_distance
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
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
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.27, random_state=69)

# probably a little cursed, but for a small model it works
with ignore_warnings(category=(ConvergenceWarning, FitFailedWarning, UserWarning)): 

    names = ['svc', 'sgd classifier', 'knn']
    classifiers = [svm.SVC(), SGDClassifier(), KNeighborsClassifier()]

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
        'max_iter':[1000, 2000, 5000],
        'epsilon':[1e-2, .1, .2, .5],
        'n_jobs':[-1],
        'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'],
        'early_stopping':[True, False]
        }
    
    knn_params = {
        'n_neighbors':[5, 10, 20, 30],
        'weights':['uniform', 'distance'],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size':[10, 30, 50, 100, 500],
        'p':[1, 2, 3],
        'metric':['minkowski', 'wasserstein_distance'],
        'n_jobs':[-1]
    }

    parameters = [svm_params, sgd_params, knn_params]

    for i in range(len(classifiers)):
        search = GridSearchCV(classifiers[i], parameters[i]).fit(X,y)
        print(f'{names[i]} achieved a peak accuracy of {round(search.best_score_, 4)}\
               \n with the parameters:{search.best_params_} \n')