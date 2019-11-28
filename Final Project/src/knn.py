from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from split_dataset import split_train_test
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_data(neighbors, MSE):
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors k')
    plt.ylabel('Misclassification Error')
    plt.show()

def run_knn(df_knn):
    print("\n\n----------------------k Nearest Neighbors----------------------\n\n")

    x = np.array(df_knn.iloc[:, 0:])
    y = np.array(df_knn['class'])

    x_train, x_test, y_train, y_test = split_train_test(x,y)

    '''neighbors = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]
    cv_scores = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors = k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]
    Accuracy = [x for x in cv_scores]
    optimal_k = neighbors[MSE.index(min(MSE))]
    optimal_k2 = neighbors[Accuracy.index(max(Accuracy))]
    print("The optimal number of neighbors is %d" % optimal_k)'''

    # try K=1 through K=31 and record testing accuracy
    k_range = range(1, 30)

    # We can create Python dictionary using [] or dict()
    scores = []

    # We use a loop through the range 1 to 31
    # We append the scores in the dictionary
    for k_val in k_range:
        knn = KNeighborsClassifier(n_neighbors=k_val)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    # print(scores)

    optimal_k_val = k_range[scores.index(max(scores))]

    print(optimal_k_val)

    classifier = KNeighborsClassifier(n_neighbors=(k_range[scores.index(max(scores))]))
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()

    '''# plot_data(neighbors, MSE)
    plot_data(neighbors, Accuracy)

    classifier = KNeighborsClassifier(n_neighbors=9)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))'''
