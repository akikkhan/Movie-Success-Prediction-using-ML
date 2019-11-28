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
                         error_bad_lines=False)

movie_df = movie_df[['id', 'budget', 'budget_log', 'genres', 'popularity', 'release_date', 'revenue',
                     'revenue_log', 'runtime', 'title', 'vote_average', 'vote_count']]
# combined_df = pd.merge(movie_df, credits_df, on='id')
dtype = dict(id=str)
combined_df = movie_df.astype(dtype).merge(credits_df.astype(dtype), 'inner')
# print(combined_df.head())
#combined_df = combined_df[(combined_df['budget'] > 1000) & (combined_df['revenue'] > 1000)]
combined_df = combined_df[(combined_df['budget']) < (combined_df['revenue'])]
combined_df = combined_df[combined_df['revenue']>1000]
combined_df_copy = combined_df[['budget', 'vote_average', 'popularity', 'vote_count']]
# print(combined_df_copy)
x = combined_df_copy
# print(len(x))

y = combined_df['revenue']


'''X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
#print(regressor.intercept_)
#For retrieving the slope:
#print(regressor.coef_)

y_pred = regressor.predict(X_test)

lin_reg_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(lin_reg_df.head())'''

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

print("The linear model has intercept : {} and coefficients : {} ".format(model.intercept_, model.coef_))
#print("Root mean square error is {} ".format(rmsle))
#print("Mean absolute error is {} ".format(maerr))

pred = np.abs(reg.predict(X_test))

mserr = mean_squared_error(pred, y_test)
print(mserr)
print(np.sqrt(mserr))
mabserr = mean_absolute_error(pred, y_test)
print(mabserr)


# y_new used to draw the best fit line
#y_new = model.intercept_ + model.coef_[0] * (X_test['budget']) + model.coef_[1] * (X_test['vote_average']) + \
#        model.coef_[2] * (X_test['popularity']) + model.coef_[3] * (X_test['vote_count'])


#print(pred[:10])
#print(y_test[:10])


# Hide the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

x_coordinate = range(1, len(pred)+1)
plt.plot(x_coordinate[:100], pred[:100], label='Predicted revenue')
plt.plot(x_coordinate[:100], y_test[:100], label='Actual revenue')
#plt.plot(x_coordinate[:100], y_new[:100], color='red', label='Best fit line')
#plt.plot(x_coordinate, pred, label='Predicted revenue')
#plt.plot(x_coordinate, y_test, label='Actual revenue')
#plt.plot(x_coordinate, y_new, color='red', label='Best fit line')
plt.xlabel('Number of records', fontsize=12, fontweight='bold')
plt.ylabel('Revenue (in 10^9)', fontsize=12, fontweight='bold')
plt.legend(loc=2)
plt.show()


'''# THIS SHOWS PLOT FOR POPULAR ACTOR REVENUE PREDICTION

combined_df_actor = combined_df[(combined_df['vote_average'] > 8.0) & (combined_df['vote_count'] > 5000)]
combined_df_actor = combined_df_actor.sort_values(['vote_average'], ascending=False)
# print(combined_df_actor)

# Analysis for actor 'Tom Hanks'
df_hanks = combined_df[(combined_df['actor_1'] == 'Tom Hanks') |
                       (combined_df['actor_2'] == ' Tom Hanks') |
                       (combined_df['actor_3'] == ' Tom Hanks') |
                       (combined_df['actor_4'] == ' Tom Hanks') |
                       (combined_df['actor_5'] == ' Tom Hanks') |
                       (combined_df['actor_6'] == ' Tom Hanks')]
df_hanks = df_hanks[(df_hanks['budget'] > 1000) & (df_hanks['revenue'] > 1000)]

df_hanks_copy = df_hanks[['budget', 'vote_average', 'popularity', 'vote_count']]
x_hanks = df_hanks_copy
y_hanks = df_hanks['revenue']


X_train_hank, X_test_hank, y_train_hank, y_test_hank = train_test_split(x_hanks, y_hanks, test_size=0.3, random_state=0)

model_hank = reg.fit(X_train_hank, y_train_hank)
y_pred_hank = reg.predict(X_train_hank)
#rmsle_hank = rmsle(y_pred_hank, y_train_hank)
#rmale_hank = rmale(y_pred_hank, y_train_hank)

#print("The linear model has intercept : {} and coefficients : {} ".format(model.intercept_, model.coef_))
#print("Root mean square error is {} ".format(rmsle_hank))
#print("Root mean absolute error is {} ".format(rmale_hank))

#pred_hank = np.abs(reg.predict(X_test_hank) - 100000000) #test prediction; subtract 10^8 if pred is greater than 10^8
                                                          # otherwise, subtract 10^7
pred_hank = np.abs(reg.predict(X_test_hank))

# y_new used to draw the best fit line
y_new_hank = model_hank.intercept_ + model_hank.coef_[0] * (X_test_hank['budget']) + \
             model_hank.coef_[1] * (X_test_hank['vote_average']) + \
             model_hank.coef_[2] * (X_test_hank['popularity']) + \
             model_hank.coef_[3] * (X_test_hank['vote_count'])

# print(pred_hank.shape)
# print(y_test_hank)
# print(y_new_hank)
# print(model_hank.intercept_)
# print(model_hank.coef_)

x_coord_hanks = range(1, len(pred_hank)+1)
plt.plot(x_coord_hanks, pred_hank, color='blue', label='Hank prediction')
plt.plot(x_coord_hanks, y_test_hank, color='orange', label='Hank actual')
#plt.plot(x_coord_hanks, y_new_hank, color='red', label='Best fit line')
plt.xlabel('Number of records')
plt.ylabel('Revenue (in hundred million or 10^8)')
plt.legend(loc=2)
plt.show()


# Analysis for actor 'Tom Cruise'
df_cruise = combined_df[(combined_df['actor_1'] == 'Tom Cruise') |
                       (combined_df['actor_2'] == ' Tom Cruise') |
                       (combined_df['actor_3'] == ' Tom Cruise') |
                       (combined_df['actor_4'] == ' Tom Cruise') |
                       (combined_df['actor_5'] == ' Tom Cruise') |
                       (combined_df['actor_6'] == ' Tom Cruise')]
df_cruise = df_cruise[(df_cruise['budget'] > 1000) & (df_cruise['revenue'] > 1000)]

df_cruise_copy = df_cruise[['budget', 'vote_average', 'popularity']]
x_cruise = df_cruise_copy
y_cruise = df_cruise['revenue']


X_train_cruise, X_test_cruise, y_train_cruise, y_test_cruise = train_test_split(x_cruise, y_cruise, test_size=0.3, random_state=0)

model_cruise = reg.fit(X_train_cruise, y_train_cruise)
y_pred_cruise = reg.predict(X_train_cruise)
#rmsle_cruise = rmsle(y_pred_cruise, y_train_cruise)
#rmale_cruise = rmale(y_pred_cruise, y_train_cruise)

#print("The linear model has intercept : {} and coefficients : {} ".format(model.intercept_, model.coef_))
#print("Root mean square error is {} ".format(rmsle_hank))
#print("Root mean absolute error is {} ".format(rmale_hank))

pred_cruise = reg.predict(X_test_cruise)

# y_new used to draw the best fit line
y_new_cruise = model_cruise.intercept_ + model_cruise.coef_[0] * (X_test_cruise['budget']) + \
             model_cruise.coef_[1] * (X_test_cruise['vote_average']) + \
             model_cruise.coef_[2] * (X_test_cruise['popularity'])

# print(pred_cruise.shape)
# print(y_test_cruise)
# print(y_new_cruise)
# print(model_cruise.intercept_)
# print(model_cruise.coef_)

x_coord_cruise = range(1, len(pred_cruise)+1)
plt.plot(x_coord_cruise, pred_cruise, color='blue', label='Cruise prediction')
plt.plot(x_coord_cruise, y_test_cruise, color='orange', label='Cruise actual')
#plt.plot(x_coord_cruise, y_new_cruise, color='red', label='Best fit line')
plt.xlabel('Number of records')
plt.ylabel('Revenue (in hundred million or 10^8)')
plt.legend(loc=1)
plt.show()


# Analysis for actor 'Morgan Freeman'
df_mfree = combined_df[(combined_df['actor_1'] == 'Morgan Freeman') |
                       (combined_df['actor_2'] == ' Morgan Freeman') |
                       (combined_df['actor_3'] == ' Morgan Freeman') |
                       (combined_df['actor_4'] == ' Morgan Freeman') |
                       (combined_df['actor_5'] == ' Morgan Freeman') |
                       (combined_df['actor_6'] == ' Morgan Freeman')]


df_mfree = df_mfree[(df_mfree['budget'] > 1000) & (df_mfree['revenue'] > 1000)]

df_mfree_copy = df_mfree[['budget', 'vote_average', 'popularity']]
x_mfree = df_mfree_copy
y_mfree = df_mfree['revenue']


X_train_mfree, X_test_mfree, y_train_mfree, y_test_mfree = train_test_split(x_mfree, y_mfree, test_size=0.3, random_state=0)

model_mfree = reg.fit(X_train_mfree, y_train_mfree)
y_pred_mfree = reg.predict(X_train_mfree)
#rmsle_mfree = rmsle(y_pred_mfree, y_train_mfree)
#rmale_mfree = rmale(y_pred_mfree, y_train_mfree)

#print("The linear model has intercept : {} and coefficients : {} ".format(model_mfree.intercept_, model_mfree.coef_))
#print("Root mean square error is {} ".format(rmsle_mfree))
#print("Root mean absolute error is {} ".format(rmale_mfree))

pred_mfree = reg.predict(X_test_mfree)

# y_new used to draw the best fit line
y_new_mfree = model_mfree.intercept_ + model_mfree.coef_[0] * (X_test_mfree['budget']) + \
             model_mfree.coef_[1] * (X_test_mfree['vote_average']) + \
             model_mfree.coef_[2] * (X_test_mfree['popularity'])

# print(pred_mfree.shape)
# print(y_test_mfree)
# print(y_new_mfree)
# print(y_pred_mfree.shape)
# print(model_mfree.intercept_)
# print(model_mfree.coef_)

x_coord_mfree = range(1, len(pred_mfree)+1)
# plt.plot(x_coord_mfree, pred_mfree, color='blue', label='Freeman prediction')
# plt.plot(x_coord_mfree, y_test_mfree, color='orange', label='Freeman actual')
# plt.plot(x_coord_mfree, y_new_mfree, color='red', label='Best fit line')
# plt.legend(loc=1)
# plt.show()'''

'''print(len(combined_df[(combined_df['actor_1'] == act_name) |
                      (combined_df['actor_2'] == act_name) |
                      (combined_df['actor_3'] == act_name) |
                      (combined_df['actor_4'] == act_name) |
                      (combined_df['actor_5'] == act_name) |
                      (combined_df['actor_6'] == act_name)]))'''

