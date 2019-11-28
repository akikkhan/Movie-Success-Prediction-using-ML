import pandas as pd
from sklearn.preprocessing import StandardScaler
from knn import run_knn
from random_forest import run_random_forest
from logistic import run_logistic_regression
from pca import run_pca
from cross_val import run_cross_val
from collections import Counter
import numpy as np

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)


def fill_nan(df_movie, col):
    df_movie[col] = df_movie[col].fillna(df_movie[col].median())


def data_prepocessing():
    df = pd.read_csv('../data/imdb.csv', error_bad_lines=False)
    df = df[df['year'] > 2000]
    df_movie = df[df['type'] != 'video.episode']

    cols = list(df_movie.columns)
    fill_nan(df_movie,cols)

    col = list(df_movie.columns)
    col.remove('type')
    col = col[5:15]

    sc = StandardScaler()
    temp = sc.fit_transform(df_movie[col])
    # df_movie[col] = temp

    df_standard = df_movie[list(df_movie.describe().columns)]
    return (df_movie, df_standard)


def classify(row):
    if row['imdbRating'] >= 0 and row['imdbRating'] < 4:
        return 0
    elif row['imdbRating'] >= 4 and row['imdbRating'] < 7:
        return 1
    elif row['imdbRating'] >= 7 and row['imdbRating'] <= 10:
        return 2


def top_keywords():
    movie_df = pd.read_csv('../data/the-movies-dataset/movies_metadata.csv', error_bad_lines=False, low_memory=False)
    key_df = pd.read_csv('../data/the-movies-dataset/keywords-edited.csv', error_bad_lines=False)
    combined_df = pd.concat([movie_df, key_df], sort=True)
    combined_df = combined_df[['id', 'keywords']].dropna()

    combined_df['keywords'] = combined_df['keywords'].apply(lambda x: x.replace('[','').replace(']',''))
    combined_df['keywords'].replace('', np.nan, inplace=True)
    combined_df.dropna(subset=['keywords'], inplace=True)


    df_only_keywords = combined_df[['keywords']]
    df_only_keywords = df_only_keywords.apply(', '.join)

    # split() returns list of all the words in the string
    split_it = df_only_keywords['keywords'].split(', ')

    # Pass the split_it list to instance of Counter class.
    #Counter = Counter(split_it)

    # most_common() produces k frequently encountered
    # input values and their respective counts.
    most_occur = Counter(split_it).most_common(10)

    print(most_occur)


def actor_frequent_info():
    movie_df = pd.read_csv(r'../data/movie_metadata_edited.csv',
                           error_bad_lines=False,
                           low_memory=False,
                           skipinitialspace=True)

    movie_df_copy = movie_df[['movie_title', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                              'genres', 'plot_keywords', 'budget', 'gross', 'imdb_score']]

    str_input = ''

    stop_words = ['quit', 'exit']

    while str_input not in stop_words:
        str_input = input('Enter type of operation: ')

        if str_input == 'title':
            movie_name = ''
            movie_name = input('Enter movie name: ')
            print(movie_df_copy.loc[movie_df_copy['movie_title'] == movie_name][['director_name',
                                                                                 'actor_1_name',
                                                                                 'actor_2_name',
                                                                                 'actor_3_name',
                                                                                 'genres']])

        if str_input == 'director':
            director_name = ''
            director_name = input('Enter director name: ')
            print(movie_df_copy.loc[movie_df_copy['director_name'] == director_name][['movie_title', 'actor_1_name',
                                                                                      'actor_2_name', 'actor_3_name',
                                                                                      'genres', 'budget', 'gross']])

        if str_input == 'actor':
            actor_name = ''
            actor_name = input('Enter actor name: ')
            print(movie_df_copy.loc[(movie_df_copy['actor_1_name'] == actor_name) |
                                    (movie_df_copy['actor_2_name'] == actor_name) |
                                    (movie_df_copy['actor_3_name'] == actor_name)][['movie_title', 'director_name',
                                                                                    'actor_1_name', 'actor_2_name',
                                                                                    'actor_3_name', 'genres',
                                                                                    'budget', 'gross']])


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # disables the SettingWithCopyError
    df_movie, df_standard = data_prepocessing()

    df_knn = df_movie
    df_knn["class"] = df_knn.apply(classify, axis=1)

    str_input = ''

    while str_input != 'quit':
        str_input = input('Enter type of operation: ')

        if str_input == 'top keywords':
            top_keywords()

        if str_input == 'actor info':
            actor_frequent_info()

        if str_input == 'pca':
            run_pca(df_standard, df_movie)

        if str_input == 'knn':
            run_knn(df_knn)

        if str_input == 'random forest':
            run_random_forest(df_knn)

        if str_input == 'logistic regression':
            run_logistic_regression()
