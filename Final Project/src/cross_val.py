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
    # print("\n\n----------------------Cross Validation----------------------\n\n")
    # Load data for x and y
    x = np.loadtxt("../data/logistic/X_train.txt")
    y = np.loadtxt("../data/logistic/y_train.txt", dtype=int)

    # Sample training set and testing set with 40% testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y)#, test_size = 0.2, random_state = 0)

    
    #X_train.shape, y_train.shape
    #X_test.shape, y_test.shape

    # 'kernel' type is "linear" for the algorithm, penalty parameter C is 1.0 by default, and wanna fit
    # training dataset for both x and y; then it prints the score
    clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    # print(clf.score(x_test, y_test))

    # 'kernel' type is "linear" for the algorithm, penalty parameter C is 1.0 by default, but this time
    # we want to calculate cross validation score using 5-fold cv (number of folds in a "(Stratified)KFold")
    clf1 = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf1, x_train, y_train, cv=5)
    # This will output 5 values since we are using 5-fold CV
    # print(scores)
    # This prints the mean score and the 95% confidence interval of the score estimate
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Preprocess (standardization, feature selection) training set data to apply on testing data to make predictions
    # StandardScaler() standardizes features by removing the mean and scaling to unit variance
    scaler = preprocessing.StandardScaler().fit(x_train)
    # It performs standardization by centering and scaling
    x_train_transformed = scaler.transform(x_train)
    clf2 = svm.SVC(C=1).fit(x_train_transformed, y_train)
    x_test_transformed = scaler.transform(x_test)
    print(clf2.score(x_test_transformed, y_test))

    '''x = np.loadtxt("../data/logistic/X_train.txt")
    y = np.loadtxt("../data/logistic/y_train.txt", dtype=int)

    x = np.array(df.iloc[:, 0:])
    y = np.array(df['class'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=1, n_estimators=250, min_samples_split=8, min_samples_leaf=4)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)
    print("Accuracy: ",accuracy_score(y_test, pred))'''

if __name__ == '__main__':
    '''df = pd.read_csv('../data/movie_metadata.csv')
    df = remove_string_cols(df)
    df = df.fillna(value=0,axis=1)
    df["class"] = df.apply(classify, axis=1)
    df = df.drop('imdb_score', 1)
    df = df.drop('facenumber_in_poster', 1)
    df = df.drop('title_year', 1)
    df = df.drop('aspect_ratio', 1)
    df = df.drop('duration', 1)
    run_random_forest(df)'''
    run_cross_val()
