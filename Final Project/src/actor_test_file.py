import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)

def actor_frequent_info():
    # Users will be able to search for specific actors/actresses, find favourite genres,
    # most frequent keywords attached to their films
    # print("My name is actor!! <-- fill in information")

    movie_df = pd.read_csv(r'../data/movie_metadata_edited.csv',
                           error_bad_lines=False,
                           low_memory=False,
                           skipinitialspace=True)
    key_df = pd.read_csv('../data/the-movies-dataset/keywords-edited.csv',
                         error_bad_lines=False)
    credits_df = pd.read_csv('../data/the-movies-dataset/credits-edited-small.csv',
                         error_bad_lines=False)

    movie_df_copy = movie_df[['movie_title', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                              'genres', 'plot_keywords', 'budget', 'gross', 'imdb_score']]

    movie_df_copy_2 = movie_df_copy
    movie_df_copy_2.set_index('genres')

    str_input = ''

    stop_loop_words = ['quit', 'exit']

    while str_input not in stop_loop_words:
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
                                                                                    'budget', 'gross',
                                                                                    'plot_keywords']])

            new_df = movie_df_copy.loc[(movie_df_copy['actor_1_name'] == actor_name) |
                                       (movie_df_copy['actor_2_name'] == actor_name) |
                                       (movie_df_copy['actor_3_name'] == actor_name)][['movie_title', 'director_name',
                                                                                       'actor_1_name', 'actor_2_name',
                                                                                       'actor_3_name', 'genres',
                                                                                       'budget', 'gross',
                                                                                       'plot_keywords']]

            # Prints top 5 genres acted by actor
            genre_df = new_df['genres'].reset_index()
            genre_df = genre_df.drop(genre_df.iloc[:, :1], axis=1)
            merge_genre_df = genre_df.apply('|'.join)
            merge_genre_df = merge_genre_df['genres'].replace('|', ' ')
            split_it = merge_genre_df.split()
            counter_var = Counter(split_it)
            most_occur = counter_var.most_common(5)
            print('Top genres worked: ')
            for genres, genres_num in most_occur:
                print(genres)

            # Prints top 5 keywords associated with the actor
            keyword_df = new_df['plot_keywords'].reset_index()
            keyword_df = keyword_df.drop(keyword_df.iloc[:, :1], axis=1)
            merge_keyword_df = keyword_df.apply('|'.join)
            merge_keyword_df = merge_keyword_df['plot_keywords'].replace('|', ' ')
            split_it_two = merge_keyword_df.split()
            counter_keywords = Counter(split_it_two)
            most_occur_keywords = counter_keywords.most_common(5)

            # remove to/for/preposition/adjectives...
            print('Top Keywords: ')
            for keywords, count in most_occur_keywords:
                if keywords not in stop_words:
                    print(keywords)


if __name__ == '__main__':
    actor_frequent_info()
