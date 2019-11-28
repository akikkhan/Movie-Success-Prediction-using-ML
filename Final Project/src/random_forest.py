from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from split_dataset import split_train_test
import numpy as np
import matplotlib.pyplot as plt

def run_random_forest(df_knn):
    print("\n\n----------------------Random Forest----------------------\n\n")

    # x = np.array(df_knn.ix[:, 0:])
    x = np.array(df_knn.iloc[:, 0:])
    y = np.array(df_knn['class'])

    x_train, x_test, y_train, y_test = split_train_test(x,y)

    rf = RandomForestClassifier(random_state=1, n_estimators=250, min_samples_split=8, min_samples_leaf=4)
    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)
    print("Accuracy: ",accuracy_score(y_test, pred))

    x_value = range(1, len(pred)+1)
    plt.plot(x_value, pred, color='b')
    plt.plot(x_value, y_test, color='y')
    plt.show()
