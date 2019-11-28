import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)
np.seterr(divide='ignore')

# Create data
movie_df = pd.read_csv(r'../data/the-movies-dataset/movies_metadata-edited - Copy.csv',
                       error_bad_lines=False,
                       low_memory=False,
                       skipinitialspace=True)
credits_df = pd.read_csv(r'../data/the-movies-dataset/credits-edited-small.csv',
                         error_bad_lines=False,
                         low_memory=False)

movie_df = movie_df[['id', 'budget', 'budget_log', 'genres', 'popularity', 'release_date', 'revenue',
                     'revenue_log', 'runtime', 'title', 'vote_average', 'vote_count']]
# combined_df = pd.merge(movie_df, credits_df, on='id')
dtype = dict(id=str)
combined_df = movie_df.astype(dtype).merge(credits_df.astype(dtype), 'inner')
# print(combined_df.head())
#combined_df = combined_df[(combined_df['budget'] > 1000) & (combined_df['revenue'] > 1000)]
combined_df = combined_df[(combined_df['budget']) < (combined_df['revenue'])]

new_df = pd.read_csv(r'../data/combinedBudRev2.csv',
                     error_bad_lines=False,
                     low_memory=False)

combined_df_copy = new_df[['budget_norm', 'vote_average_norm', 'popularity_norm', 'vote_count_norm']]

x = combined_df_copy
# print(len(x))


y = new_df['revenue_norm']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
reg = LinearRegression()


def rmsle(y_train, y0):
    return np.sqrt(np.mean(np.square(y_train - y0)))


def maerr(y_train, y0):
    return np.mean(np.abs(y_train - y0))


model = reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)
rmsle = rmsle(y_pred, y_train)
maerr = maerr(y_pred, y_train)

#model_1 = reg.fit(X_test, y_test)
#y_pred_1 = reg.predict(X_test)
#serr = rmsle(y_pred_1, y_train)
#abserr = rmsle(y_pred_1, y_train)


#print("The linear model has intercept : {} and coefficients : {} ".format(model.intercept_, model.coef_))
#print("Root mean square error is {} ".format(rmsle))
#print("Mean absolute error is {} ".format(maerr))

#print(rmsle[5:15])
#print(maerr[5:15])

pred = np.abs(reg.predict(X_test))

mserr = mean_squared_error(pred, y_test)
#print(mserr)
#print(np.sqrt(mserr))
mabserr = mean_absolute_error(pred, y_test)
#print(mabserr)

#print(pred[:5])
#print(y_test[:5])

predict = []
for i in range(0, len(pred)):
    predict = (pred * (max(new_df['revenue']) - min(new_df['revenue']))) + min(new_df['revenue'])
    i += 1

#print(predict[:5])

#print(y_test[:5])
actual = []
for i in range(0, len(y_test)):
    actual = (y_test * (max(new_df['revenue']) - min(new_df['revenue']))) + min(new_df['revenue'])
    i += 1

#print(actual[:5])

# y_new used to draw the best fit line
#y_new = model.intercept_ + model.coef_[0] * (X_test['budget_norm']) + model.coef_[1] * (X_test['vote_average_norm']) + \
#        model.coef_[2] * (X_test['popularity_norm']) + model.coef_[3] * (X_test['vote_count_norm'])


#abs_err = mean_absolute_error(y_new, y_test)

actual_abs_error = (mabserr * (max(new_df['revenue']) - min(new_df['revenue']))) + min(new_df['revenue'])
print(actual_abs_error)

actual_ms_error = (mserr * (max(new_df['revenue']) - min(new_df['revenue']))) + min(new_df['revenue'])
print(actual_ms_error)

#mae = mean_absolute_error(predict, actual)
#print(mae)

x_coordinate = range(1, len(predict)+1)
plt.plot(x_coordinate[:100], predict[:100], label='Predicted revenue')
plt.plot(x_coordinate[:100], actual[:100], label='Actual revenue')
#plt.plot(x_coordinate[:100], y_new[:100], color='red', label='Best fit line')
#plt.plot(x_coordinate, pred, label='Predicted revenue')
#plt.plot(x_coordinate, y_test, label='Actual revenue')
#plt.plot(x_coordinate, y_new, color='red', label='Best fit line')
plt.xlabel('Number of records')
plt.ylabel('Revenue (in 10^9)')
plt.legend(loc=2)
#plt.show()

#print(pred[:5])
#print(predict[:5])
