import timeit

cross_val_code_test = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import datasets
from sklearn import svm
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def run_cross_val():
    print("----------------------Cross Validation----------------------")
    x = np.loadtxt("../data/logistic/X_train.txt")
    y = np.loadtxt("../data/logistic/y_train.txt", dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)    

    clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

    clf1 = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf1, x_train, y_train, cv=5)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_transformed = scaler.transform(x_train)
    clf2 = svm.SVC(C=1).fit(x_train_transformed, y_train)
    x_test_transformed = scaler.transform(x_test)
    print(clf2.score(x_test_transformed, y_test))

    
if __name__ == '__main__':    
    run_cross_val()
"""
speed = timeit.timeit(cross_val_code_test, number=100) / 100
print(speed)
