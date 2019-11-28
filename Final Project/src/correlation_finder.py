import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)
np.seterr(divide = 'ignore')

movie_df = pd.read_csv(r'../data/the-movies-dataset/movies_metadata-edited.csv',
                       error_bad_lines=False,
                       low_memory=False,
                       skipinitialspace=True)
credits_df = pd.read_csv(r'../data/the-movies-dataset/credits-edited-small.csv',
                         error_bad_lines=False)

movie_df = movie_df[['id', 'budget', 'budget_log', 'genres', 'popularity', 'release_date', 'revenue',
                     'revenue_log', 'runtime', 'title', 'vote_average', 'vote_count']]
movie_df_copy = movie_df[(movie_df['budget_log'] > 0) &
                         (movie_df['revenue_log'] > 0) &
                         (movie_df['runtime'] > 0) &
                         (movie_df['vote_count'] > 10)].reset_index(drop=True) # drop=True to avoid old index
#print(movie_df_copy.head())


'''corr1, _ = pearsonr(movie_df_copy['budget_log'], movie_df_copy['revenue_log'])
corr2, _ = pearsonr(movie_df_copy['vote_average'], movie_df_copy['revenue_log'])
corr3, _ = pearsonr(movie_df_copy['popularity'], movie_df_copy['revenue_log'])
corr4, _ = pearsonr(movie_df_copy['runtime'], movie_df_copy['revenue_log'])
corr5, _ = pearsonr(movie_df_copy['vote_count'], movie_df_copy['revenue_log'])

print("Pearson's correlation between budget and revenue: {0}".format(corr1))
print("Pearson's correlation between vote_average and revenue: {0}".format(corr2))
print("Pearson's correlation between popularity and revenue: {0}".format(corr3))
print("Pearson's correlation between runtime and revenue: {0}".format(corr4))
print("Pearson's correlation between vote_count and revenue: {0}".format(corr5))'''


df_movie_date = movie_df
df_movie_date['release_date'] = pd.to_datetime(df_movie_date['release_date'])
#print(df_movie_date[df_movie_date['release_date'] > '2017-12-31'])

'''datetime.datetime.now().date()
datetime.date(2015, 10, 28)
tmp = '2015-10-28 16:09:59'
dt = datetime.datetime.strptime(tmp,'%Y-%m-%d %H:%M:%S')
dt.date()
datetime.date(2015, 10, 28)
dd = dt.date()
print dd
2015-10-28'''


dtype = dict(id=str)
combined_df = movie_df.astype(dtype).merge(credits_df.astype(dtype), 'inner')
# print(combined_df.head())
combined_df = combined_df[(combined_df['budget'] > 1000) & (combined_df['revenue'] > 1000)]


#print(len(combined_df['actor_1'].unique()))

'''# FIND MOST OCCURRED ACTORS IN actor_1 COLUMN
combined_df = combined_df[['id', 'actor_1']].dropna()

combined_df['actor_1'].replace('', np.nan, inplace=True)
combined_df.dropna(subset=['actor_1'], inplace=True)

df_only_actor_1 = combined_df[['actor_1']]
df_only_actor_1 = df_only_actor_1.apply(', '.join)

# split() returns list of all the words in the string
split_it = df_only_actor_1['actor_1'].split(', ')

# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter(split_it).most_common(10)

print(most_occur)

# OUTPUT: [('John Wayne', 94), ('Jackie Chan', 73), ('Nicolas Cage', 60), ('Robert De Niro', 55), 
# ('Gérard Depardieu', 52), ('Burt Lancaster', 50), ('Michael Caine', 50), ('Bruce Willis', 47), ('Paul Newman', 47), 
# ('Jeff Bridges', 46)]'''

#combined_df_copy = combined_df[combined_df['revenue'] > combined_df['budget']]
#print(len(combined_df_copy))
combined_df_2 = combined_df[combined_df['revenue'] > combined_df['budget']]
combined_df_copy = combined_df_2[['budget', 'vote_average', 'popularity', 'vote_count']]

x = combined_df_copy

y = combined_df_2['revenue']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
reg = LinearRegression()


def rmsle(y_train, y0):
    return ((np.square(y_train - y0)))


def maerr(y_train, y0):
    return np.square(np.abs(y_train - y0))


model = reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)
rmsle = rmsle(y_pred, y_train)
maerr = maerr(y_pred, y_train)

#print("The linear model has intercept : {} and coefficients : {} ".format(model.intercept_, model.coef_))
#print("Root mean square error is {} ".format(rmsle))
#print("Mean absolute error is {} ".format(maerr))


pred = reg.predict(X_test)

# y_new used to draw the best fit line
y_new = model.intercept_ + model.coef_[0] * (X_test['budget']) + model.coef_[1] * (X_test['vote_average']) + \
        model.coef_[2] * (X_test['popularity']) + model.coef_[3] * (X_test['vote_count'])



x_coordinate = range(1, len(pred)+1)
plt.plot(x_coordinate[:100], pred[:100], label='Predicted revenue')
plt.plot(x_coordinate[:100], y_test[:100], label='Actual revenue')

plt.xlabel('Number of records')
plt.ylabel('Revenue')
plt.legend(loc=2)
plt.show()



'''df_cage = combined_df[(combined_df['actor_1'] == 'Nicolas Cage') |
                       (combined_df['actor_2'] == ' Nicolas Cage') |
                       (combined_df['actor_3'] == ' Nicolas Cage') |
                       (combined_df['actor_4'] == ' Nicolas Cage') |
                       (combined_df['actor_5'] == ' Nicolas Cage') |
                       (combined_df['actor_6'] == ' Nicolas Cage')]

df_cage = df_cage[(df_cage['budget'] > 1000) & (df_cage['revenue'] > 1000)]

df_cage_copy = df_cage[['budget', 'vote_average', 'popularity', 'vote_count']]
x_cage = df_cage_copy
y_cage = df_cage['revenue']

reg = LinearRegression()

X_train_cage, X_test_cage, y_train_cage, y_test_cage = train_test_split(x_cage, y_cage, test_size=0.3, random_state=0)

model_cage = reg.fit(X_train_cage, y_train_cage)
y_pred_cage = reg.predict(X_train_cage)

pred_cage = reg.predict(X_test_cage)

# y_new used to draw the best fit line
y_new_hank = model_cage.intercept_ + model_cage.coef_[0] * (X_test_cage['budget']) + \
             model_cage.coef_[1] * (X_test_cage['vote_average']) + \
             model_cage.coef_[2] * (X_test_cage['popularity']) + \
             model_cage.coef_[3] * (X_test_cage['vote_count'])


x_coord_cage = range(1, len(pred_cage)+1)
plt.plot(x_coord_cage, pred_cage, color='blue', label='Cage prediction')
plt.plot(x_coord_cage, y_test_cage, color='orange', label='Cage actual')
#plt.plot(x_coord_hanks, y_new_hank, color='red', label='Best fit line')
plt.xlabel('Number of records')
plt.ylabel('Revenue')
plt.legend(loc=2)
plt.show()'''
