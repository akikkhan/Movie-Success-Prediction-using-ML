import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)
np.seterr(divide = 'ignore')

# Create data
movie_df = pd.read_csv(r'../data/the-movies-dataset/movies_metadata-edited.csv',
                       error_bad_lines=False,
                       low_memory=False,
                       skipinitialspace=True)
credits_df = pd.read_csv(r'../data/the-movies-dataset/credits-edited-small.csv',
                         error_bad_lines=False)

movie_df = movie_df[['id', 'budget', 'budget_log', 'genres', 'popularity', 'release_date', 'revenue',
                     'revenue_log', 'runtime', 'title', 'vote_average', 'vote_count']]

dtype = dict(id=str)
combined_df = movie_df.astype(dtype).merge(credits_df.astype(dtype), 'inner')

combined_df_copy = combined_df[['budget', 'vote_average', 'popularity']]


def rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y)-np.log1p(y0))))


def rmale(y, y0):
    return np.sqrt(np.mean(np.abs(np.log1p(y)-np.log1p(y0))))


# define functions mse and mae, without mean and sqrt. look at the article for suggestion
def mserr(y, y0):
    return np.mean(np.square(y-y0))

combined_df = combined_df[(combined_df['budget'] >=0) & (combined_df['revenue'] >= 0)]
x = combined_df_copy
y = combined_df['revenue']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
reg = LinearRegression()


model = reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)
rmsle = rmsle(y_pred, y_train)
rmale = rmale(y_pred, y_train)
mserr = mserr(y_pred, y_train)

#y_new = model.intercept_ + model.coef_[0] * (combined_df['budget']) + model.coef_[1] * (combined_df['vote_average']) + \
#        model.coef_[1] * (combined_df['popularity'])

# instead of reg.predict, we will use this formula and use these values as our predicted values. then we can
# plot y_new against y_test to see how different our prediction is.
y_new = model.intercept_ + model.coef_[0] * (X_test['budget']) + model.coef_[1] * (X_test['vote_average']) + \
        model.coef_[1] * (X_test['popularity'])
pred = reg.predict(X_test)
#print(y_new.head(2))
#print(y_test.head(2))
#print(pred[1])
#print(y_pred[1])

#print(mean_squared_error(y_test, pred))
#print(r2_score(y_test, pred))
#print(mserr)

# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
plt.plot(y_test, pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
