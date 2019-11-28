import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

new_df = pd.read_csv(r'../data/testKeyword.csv',
                     error_bad_lines=False,
                     low_memory=False,
                     skipinitialspace=True)

movie_df = movie_df[['id', 'budget', 'budget_log', 'genres', 'popularity', 'release_date', 'revenue',
                     'revenue_log', 'runtime', 'title', 'vote_average', 'vote_count']]

# combined_df = pd.merge(movie_df, credits_df, on='id')
dtype = dict(id=str)
combined_df = movie_df.astype(dtype).merge(credits_df.astype(dtype), 'inner')
# print(combined_df.head())
movie_df_copy = combined_df[(combined_df['budget'] > 1000) & (combined_df['revenue'] > 1000)].reset_index(drop=True)


#movie_df_copy = movie_df[(movie_df['budget'] > 0) & (movie_df['revenue'] > 0)].reset_index(drop=True) # drop=True to avoid old index
#print(movie_df_copy.head())

# List of genres
# Action, Adventure, Animation, Comedy, Crime, Documentary,  Drama, Family, Fantasy, Foreign,
# History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, TV Movie, War, Western

action_df = new_df[(new_df['genre1'] == 'Action') |
                   (new_df['genre2'] == 'Action') |
                   (new_df['genre3'] == 'Action') |
                   (new_df['genre4'] == 'Action') |
                   (new_df['genre5'] == 'Action') |
                   (new_df['genre6'] == 'Action') |
                   (new_df['genre7'] == 'Action') |
                   (new_df['genre8'] == 'Action')]
adventure_df = new_df[(new_df['genre1'] == 'Adventure') |
                      (new_df['genre2'] == 'Adventure') |
                      (new_df['genre3'] == 'Adventure') |
                      (new_df['genre4'] == 'Adventure') |
                      (new_df['genre5'] == 'Adventure') |
                      (new_df['genre6'] == 'Adventure') |
                      (new_df['genre7'] == 'Adventure') |
                      (new_df['genre8'] == 'Adventure')]
animation_df = new_df[(new_df['genre1'] == 'Animation') |
                      (new_df['genre2'] == 'Animation') |
                      (new_df['genre3'] == 'Animation') |
                      (new_df['genre4'] == 'Animation') |
                      (new_df['genre5'] == 'Animation') |
                      (new_df['genre6'] == 'Animation') |
                      (new_df['genre7'] == 'Animation') |
                      (new_df['genre8'] == 'Animation')]
comedy_df = new_df[(new_df['genre1'] == 'Comedy') |
                   (new_df['genre2'] == 'Comedy') |
                   (new_df['genre3'] == 'Comedy') |
                   (new_df['genre4'] == 'Comedy') |
                   (new_df['genre5'] == 'Comedy') |
                   (new_df['genre6'] == 'Comedy') |
                   (new_df['genre7'] == 'Comedy') |
                   (new_df['genre8'] == 'Comedy')]
crime_df = new_df[(new_df['genre1'] == 'Crime') |
                   (new_df['genre2'] == 'Crime') |
                   (new_df['genre3'] == 'Crime') |
                   (new_df['genre4'] == 'Crime') |
                   (new_df['genre5'] == 'Crime') |
                   (new_df['genre6'] == 'Crime') |
                   (new_df['genre7'] == 'Crime') |
                   (new_df['genre8'] == 'Crime')]
documentary_df = new_df[(new_df['genre1'] == 'Documentary') |
                   (new_df['genre2'] == 'Documentary') |
                   (new_df['genre3'] == 'Documentary') |
                   (new_df['genre4'] == 'Documentary') |
                   (new_df['genre5'] == 'Documentary') |
                   (new_df['genre6'] == 'Documentary') |
                   (new_df['genre7'] == 'Documentary') |
                   (new_df['genre8'] == 'Documentary')]
drama_df = new_df[(new_df['genre1'] == 'Drama') |
                   (new_df['genre2'] == 'Drama') |
                   (new_df['genre3'] == 'Drama') |
                   (new_df['genre4'] == 'Drama') |
                   (new_df['genre5'] == 'Drama') |
                   (new_df['genre6'] == 'Drama') |
                   (new_df['genre7'] == 'Drama') |
                   (new_df['genre8'] == 'Drama')]
family_df = new_df[(new_df['genre1'] == 'Family') |
                   (new_df['genre2'] == 'Family') |
                   (new_df['genre3'] == 'Family') |
                   (new_df['genre4'] == 'Family') |
                   (new_df['genre5'] == 'Family') |
                   (new_df['genre6'] == 'Family') |
                   (new_df['genre7'] == 'Family') |
                   (new_df['genre8'] == 'Family')]
fantasy_df = new_df[(new_df['genre1'] == 'Fantasy') |
                   (new_df['genre2'] == 'Fantasy') |
                   (new_df['genre3'] == 'Fantasy') |
                   (new_df['genre4'] == 'Fantasy') |
                   (new_df['genre5'] == 'Fantasy') |
                   (new_df['genre6'] == 'Fantasy') |
                   (new_df['genre7'] == 'Fantasy') |
                   (new_df['genre8'] == 'Fantasy')]
foreign_df = new_df[(new_df['genre1'] == 'Foreign') |
                   (new_df['genre2'] == 'Foreign') |
                   (new_df['genre3'] == 'Foreign') |
                   (new_df['genre4'] == 'Foreign') |
                   (new_df['genre5'] == 'Foreign') |
                   (new_df['genre6'] == 'Foreign') |
                   (new_df['genre7'] == 'Foreign') |
                   (new_df['genre8'] == 'Foreign')]
history_df = new_df[(new_df['genre1'] == 'History') |
                   (new_df['genre2'] == 'History') |
                   (new_df['genre3'] == 'History') |
                   (new_df['genre4'] == 'History') |
                   (new_df['genre5'] == 'History') |
                   (new_df['genre6'] == 'History') |
                   (new_df['genre7'] == 'History') |
                   (new_df['genre8'] == 'History')]
horror_df = new_df[(new_df['genre1'] == 'Horror') |
                   (new_df['genre2'] == 'Horror') |
                   (new_df['genre3'] == 'Horror') |
                   (new_df['genre4'] == 'Horror') |
                   (new_df['genre5'] == 'Horror') |
                   (new_df['genre6'] == 'Horror') |
                   (new_df['genre7'] == 'Horror') |
                   (new_df['genre8'] == 'Horror')]
music_df = new_df[(new_df['genre1'] == 'Music') |
                   (new_df['genre2'] == 'Music') |
                   (new_df['genre3'] == 'Music') |
                   (new_df['genre4'] == 'Music') |
                   (new_df['genre5'] == 'Music') |
                   (new_df['genre6'] == 'Music') |
                   (new_df['genre7'] == 'Music') |
                   (new_df['genre8'] == 'Music')]
mystery_df = new_df[(new_df['genre1'] == 'Mystery') |
                   (new_df['genre2'] == 'Mystery') |
                   (new_df['genre3'] == 'Mystery') |
                   (new_df['genre4'] == 'Mystery') |
                   (new_df['genre5'] == 'Mystery') |
                   (new_df['genre6'] == 'Mystery') |
                   (new_df['genre7'] == 'Mystery') |
                   (new_df['genre8'] == 'Mystery')]
romance_df = new_df[(new_df['genre1'] == 'Romance') |
                   (new_df['genre2'] == 'Romance') |
                   (new_df['genre3'] == 'Romance') |
                   (new_df['genre4'] == 'Romance') |
                   (new_df['genre5'] == 'Romance') |
                   (new_df['genre6'] == 'Romance') |
                   (new_df['genre7'] == 'Romance') |
                   (new_df['genre8'] == 'Romance')]
scifi_df = new_df[(new_df['genre1'] == 'Science Fiction') |
                   (new_df['genre2'] == 'Science Fiction') |
                   (new_df['genre3'] == 'Science Fiction') |
                   (new_df['genre4'] == 'Science Fiction') |
                   (new_df['genre5'] == 'Science Fiction') |
                   (new_df['genre6'] == 'Science Fiction') |
                   (new_df['genre7'] == 'Science Fiction') |
                   (new_df['genre8'] == 'Science Fiction')]
thriller_df = new_df[(new_df['genre1'] == 'Thriller') |
                   (new_df['genre2'] == 'Thriller') |
                   (new_df['genre3'] == 'Thriller') |
                   (new_df['genre4'] == 'Thriller') |
                   (new_df['genre5'] == 'Thriller') |
                   (new_df['genre6'] == 'Thriller') |
                   (new_df['genre7'] == 'Thriller') |
                   (new_df['genre8'] == 'Thriller')]
tvmovie_df = new_df[(new_df['genre1'] == 'TV Movie') |
                   (new_df['genre2'] == 'TV Movie') |
                   (new_df['genre3'] == 'TV Movie') |
                   (new_df['genre4'] == 'TV Movie') |
                   (new_df['genre5'] == 'TV Movie') |
                   (new_df['genre6'] == 'TV Movie') |
                   (new_df['genre7'] == 'TV Movie') |
                   (new_df['genre8'] == 'TV Movie')]
war_df = new_df[(new_df['genre1'] == 'War') |
                   (new_df['genre2'] == 'War') |
                   (new_df['genre3'] == 'War') |
                   (new_df['genre4'] == 'War') |
                   (new_df['genre5'] == 'War') |
                   (new_df['genre6'] == 'War') |
                   (new_df['genre7'] == 'War') |
                   (new_df['genre8'] == 'War')]
western_df = new_df[(new_df['genre1'] == 'Western') |
                   (new_df['genre2'] == 'Western') |
                   (new_df['genre3'] == 'Western') |
                   (new_df['genre4'] == 'Western') |
                   (new_df['genre5'] == 'Western') |
                   (new_df['genre6'] == 'Western') |
                   (new_df['genre7'] == 'Western') |
                   (new_df['genre8'] == 'Western')]

types= ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Docu',  'Drama', 'Family', 'Fantasy', 'Foreign',
        'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'TV Movie', 'War', 'Western']

x = [action_df['budget'].mean(), adventure_df['budget'].mean(), animation_df['budget'].mean(), comedy_df['budget'].mean()
     , crime_df['budget'].mean(), documentary_df['budget'].mean(), drama_df['budget'].mean(), family_df['budget'].mean()
     , fantasy_df['budget'].mean(), foreign_df['budget'].mean(), history_df['budget'].mean(), horror_df['budget'].mean()
     , music_df['budget'].mean(), mystery_df['budget'].mean(), romance_df['budget'].mean(), scifi_df['budget'].mean()
     , thriller_df['budget'].mean(), tvmovie_df['budget'].mean(), war_df['budget'].mean(), western_df['budget'].mean()]
y = [action_df['revenue'].mean(), adventure_df['revenue'].mean(), animation_df['revenue'].mean()
    , comedy_df['revenue'].mean(), crime_df['revenue'].mean(), documentary_df['revenue'].mean()
    , drama_df['revenue'].mean(), family_df['revenue'].mean(), fantasy_df['revenue'].mean()
    , foreign_df['revenue'].mean(), history_df['revenue'].mean(), horror_df['revenue'].mean()
    , music_df['revenue'].mean(), mystery_df['revenue'].mean(), romance_df['revenue'].mean()
    , scifi_df['revenue'].mean(), thriller_df['revenue'].mean(), tvmovie_df['revenue'].mean()
    , war_df['revenue'].mean(), western_df['revenue'].mean()]

#print(x.index(max(x)))

'''for i, type in enumerate(types):
    x_coord = x[i]
    y_coord = y[i]
    plt.scatter(x_coord, y_coord, marker='o', color='red')
    plt.text(x_coord+0.3, y_coord+0.3, type, fontsize=9)
plt.xlabel('Movie budget')
plt.ylabel('Movie revenue')
# Hide the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()'''

# Stacked bar chart with legend
Margins = ['Budget', 'Profit']
#pos = np.arange(len(types))
# Happiness_Index_Male = [60, 40, 70, 65, 85] <-- budget i.e. x
# Happiness_Index_Female = [30, 60, 70, 55, 75] <-- revenue i.e. y

'''plt.bar(types, x, color='orange', edgecolor='black')
plt.bar(types, y, color='blue', edgecolor='black', bottom=x)
plt.xticks(types, fontsize=8, rotation=30)
plt.xlabel('Genres', fontsize=12)
plt.ylabel('Currency', fontsize=12)
plt.title('Stacked Barchart - Budget and Revenue for movies', fontsize=14)
plt.legend(Margins, loc=1)
plt.show()'''


'''plt.bar(types, y, label='Revenue', alpha=0.9)
#plt.xlabel('Genre', fontsize=5, fontweight='bold')
#plt.ylabel('Revenue', fontsize=5, fontweight='bold')
plt.xticks(types, fontsize=8, rotation=30)
#plt.title('Revenue for each Genre')
plt.bar(types, x, label='Budget', alpha=0.9)
plt.xlabel('Genre', fontsize=10, fontweight='bold')
plt.ylabel('Amount in dollars', fontsize=10, fontweight='bold')
# Hide the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xticks(types, fontsize=8, rotation=30)
plt.title('Budget and Revenue for each Genre')
plt.legend()
plt.show()'''

# The position of the bars on the x-axis
r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Create first bars
#plt.bar(r, x, label='Budget', alpha=0.9)
# Create green bars (middle), on top of the firs ones
#plt.bar(r, y, bottom=x, label='Revenue', alpha=0.9)
plt.bar(r, y, color='#FF8627', label='Revenue', alpha=0.9)

# Custom X axis
plt.xticks(r, types, fontsize=8, rotation=30)
plt.xlabel("Genre", fontsize=12, fontweight='bold')
plt.ylabel('Revenue (in 10^8 USD)', fontsize=12, fontweight='bold')
#plt.legend()

# Hide the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Show graphic
plt.show()

# print(movie_df.vote_average.unique())
'''[ 7.7  6.9  6.5  6.1  5.7  6.2  5.4  5.5  6.6  7.1  7.8  7.2  6.4  6.   6.3  7.   7.4  7.6  6.8  7.3  3.5  6.7  8.1  5.9
  5.2  3.   5.8  4.5  4.4  2.8  4.1  5.1  3.9  7.5  0.   7.9  5.6  3.3  5.3  4.3  3.8  5.   4.  10.   4.9  4.6  4.7  2.5
  4.8  8.2  8.3  8.5  8.   2.   3.4  3.7  4.2  3.6  2.7  3.2  2.9  9.   9.3  8.8  8.7  1.5  1.7  3.1  1.   8.4  2.4  8.6
  8.9  1.2  1.6  2.3  1.3  1.9  0.5  2.1  2.6  9.1  1.8  9.5  9.2  9.6  2.2  1.4  9.8  9.4  0.7  1.1]'''

movie_df_small_graph = movie_df
movie_df_small_graph['vote_average_ceil'] = np.ceil(movie_df_small_graph['vote_average'])

bar_size = []
bar_percent = []
total_vote = movie_df_small_graph['vote_count'].sum()

for ratings in range(0, 11):
    movie_df_small_graph_each = movie_df_small_graph[movie_df_small_graph['vote_average_ceil'] == ratings]
    x = movie_df_small_graph_each['vote_count'].sum()
    bar_size.append(x)

for nums in range(len(bar_size)):
    perc_value = bar_size[nums] / total_vote * 100
    bar_percent.append(perc_value)

#print(bar_size)
# [254, 159, 742, 4878, 24695, 155915, 979761, 1986867, 1623719, 217810, 1133]
#my_formatted_list = [ "%.3f" % elem for elem in bar_percent ]
#print(my_formatted_list)
#print(bar_percent)
#print(len(movie_df_small_graph_0))

# print(movie_df.columns)

bars = plt.bar(range(0, 11), bar_size, alpha=0.8)


plt.xticks(range(0, 11))
#for spine in plt.gca().spines.values():
 #   spine.set_visible(False)
# Hide the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
#plt.yticks([])
plt.ylim(0, 2000000)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True)

plt.xlabel('Ratings', fontsize=12, alpha=0.9)
plt.ylabel('Vote count', fontsize=12, alpha=0.9)
plt.title('Rating distribution', fontsize=12, alpha=0.9)

for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(float("%.3f" % (bar.get_height()/total_vote*100))) + '%',
                   ha='center', color='r', fontsize=11, alpha=0.8)

plt.show()