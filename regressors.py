import pandas as pd
import numpy as np
from sklearn import svm
from scipy.stats import wasserstein_distance
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