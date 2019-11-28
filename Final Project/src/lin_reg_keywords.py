import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from collections import Counter

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)
np.seterr(divide='ignore')

movie_df = pd.read_csv('../data/combinedBudRev2.csv', error_bad_lines=False, low_memory=False)
key_df = pd.read_csv('../data/the-movies-dataset/keywords-edited.csv', error_bad_lines=False)

test_df = pd.read_csv('../data/testKeyword.csv', error_bad_lines=False)

action_key_df = test_df[(test_df['genre1'] == 'Action') |
                        (test_df['genre2'] == 'Action') |
                        (test_df['genre3'] == 'Action') |
                        (test_df['genre4'] == 'Action') |
                        (test_df['genre5'] == 'Action') |
                        (test_df['genre6'] == 'Action') |
                        (test_df['genre7'] == 'Action') |
                        (test_df['genre8'] == 'Action')]

# FIND MOST OCCURRED ACTORS IN actor_1 COLUMN
act_key_df = action_key_df[['keywords']].dropna()

act_key_df['keywords'] = act_key_df['keywords'].apply(lambda x: x.replace('[' ,'').replace(']' ,''))
act_key_df['keywords'].replace('', np.nan, inplace=True)
act_key_df.dropna(subset=['keywords'], inplace=True)

df_only_words = act_key_df[['keywords']]
df_only_words = df_only_words.apply(', '.join)

# split() returns list of all the words in the string
split_it = df_only_words['keywords'].split(', ')

# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter(split_it).most_common(12)

print(most_occur)

'''combined_df = pd.concat([movie_df, key_df], sort=True)
combined_df = combined_df[['id', 'keywords']].dropna()

combined_df['keywords'] = combined_df['keywords'].apply(lambda x: x.replace('[' ,'').replace(']' ,''))
combined_df['keywords'].replace('', np.nan, inplace=True)
combined_df.dropna(subset=['keywords'], inplace=True)


df_only_keywords = combined_df[['keywords']]
df_only_keywords = df_only_keywords.apply(', '.join)

# split() returns list of all the words in the string
split_it = df_only_keywords['keywords'].split(', ')


# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter(split_it).most_common(15)

print(most_occur)'''


'''# Create data
movie_df = pd.read_csv(r'../data/the-movies-dataset/movies_metadata-edited - Copy.csv',
                       error_bad_lines=False,
                       low_memory=False,
                       skipinitialspace=True)
credits_df = pd.read_csv(r'../data/the-movies-dataset/credits-edited-small.csv',
                         error_bad_lines=False)

movie_df = movie_df[['id', 'budget', 'budget_log', 'genres', 'popularity', 'release_date', 'revenue',
                     'revenue_log', 'runtime', 'title', 'vote_average', 'vote_count']]
# combined_df = pd.merge(movie_df, credits_df, on='id')
dtype = dict(id=str)
combined_df = movie_df.astype(dtype).merge(credits_df.astype(dtype), 'inner')
# print(combined_df.head())
#combined_df = combined_df[(combined_df['budget'] > 1000) & (combined_df['revenue'] > 1000)]
combined_df = combined_df[(combined_df['budget']) < (combined_df['revenue'])]

new_df = pd.read_csv(r'../data/combinedBudRev2.csv', error_bad_lines=False)'''

combined_df_copy = test_df[['budget', 'vote_average', 'popularity', 'vote_count']]#, 'murder', 'violence',
#                            'biography', 'revenge', 'suspense', 'love']]

#[['budget', 'vote_average', 'popularity', 'vote_count', 'woman director',
#'independent film', 'murder', 'based on novel', 'musical', 'sex', 'violence', 'nudity',
#'biography', 'revenge', 'suspense', 'love', 'female nudity', 'sport', 'police']]

x = combined_df_copy
# print(len(x))


y = test_df['revenue']


'''# FIND MOST OCCURRED ACTORS IN actor_1 COLUMN
combined_df_dir = combined_df[['director']].dropna()

combined_df_dir['director'].replace('', np.nan, inplace=True)
combined_df_dir.dropna(subset=['director'], inplace=True)

df_only_actor_1 = combined_df_dir[['director']]
df_only_actor_1 = df_only_actor_1.apply(', '.join)

# split() returns list of all the words in the string
split_it = df_only_actor_1['director'].split(', ')

# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter(split_it).most_common(20)

print(most_occur)'''


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


#print("The linear model has intercept : {} and coefficients : {} ".format(model.intercept_, model.coef_))
#print("Root mean square error is {} ".format(rmsle))
#print("Mean absolute error is {} ".format(maerr))

#print(rmsle[5:15])
#print(maerr[5:15])

pred = np.abs(reg.predict(X_test))

mserr = mean_squared_error(pred, y_test)
print(mserr)
print(np.sqrt(mserr))
mabserr = mean_absolute_error(pred, y_test)
print(mabserr)

#print(pred[:5])
#print(y_test[:5])

'''predict = []
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

print(max(new_df['revenue']))
print(min(new_df['revenue']))

actual_abs_error = (mabserr * (max(new_df['revenue']) - min(new_df['revenue']))) + min(new_df['revenue'])
print(actual_abs_error)

#mae = mean_absolute_error(predict, actual)
#print(mae)'''

x_coordinate = range(1, len(pred)+1)
plt.plot(x_coordinate[:100], pred[:100], label='Predicted revenue')
plt.plot(x_coordinate[:100], y_test[:100], label='Actual revenue')

plt.xlabel('Number of records')
plt.ylabel('Revenue (in 10^9)')
plt.legend(loc=2)
plt.show()

