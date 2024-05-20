import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
 
# 1 = heart disease
# 0 = healthy

df = pd.read_csv('data.csv')
categories = list(df.columns)
X = df.drop('target', axis=1)
y = df.iloc[:,-1:].values.ravel()

models = [

]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=69)

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))









