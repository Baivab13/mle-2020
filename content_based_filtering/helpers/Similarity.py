from content_based_filtering.helpers.movies import *
from content_based_filtering.helpers.Users import *

import numpy as np
import pandas as pd


def get_movies_similarity_matrix(movies_genre):
    
    return movies_genre.values.dot(movies_genre.values.T)

def get_users_similarity_matrix(users_info):
    
    entry_matrix=users_info.values
    similarity_matrix=np.empty([entry_matrix.shape[0],entry_matrix.shape[0]])
    for i in range(entry_matrix.shape[0]):
        for j in range(entry_matrix.shape[0]):
            similarity_matrix[i][j]= np.sum(abs(entry_matrix[i]-entry_matrix[j]))
            
    return similarity_matrix

def get_most_similar_movies(movies_similarity, movies, movie_name, year=None, top=10):
    
    index_movie = get_movie_id(movies, movie_name, year)  
    best = movies_similarity[index_movie].sort_values(ascending=False).index
    return [(ind, get_movie_name(movies, ind), movies_similarity[index_movie, ind]) for ind in best[:top] if ind != index_movie]

def get_most_similar_users(user_similarity,users,user_id,top=10):
    
    index_user=get_user_index(users,user_id)
    best = (-1*user_similarity[index_user]).sort_values(ascending=False).index
    return [(ind, get_user_ID(users, ind), user_similarity[index_user, ind]) for ind in best[:top] if ind != index_user]
    

def get_content_based_recommendations(dataframe,movies,movies_similarity,user_id,rating_col='rating',user_id_col='user_id',
                                      movie_id_col='movie_id',top=10,nb_recommendations=5):
    
    top_movies = dataframe[dataframe[user_id_col] == user_id].sort_values(by=rating_col,ascending=False).head(top)[movie_id_col]
    index=['movie_id', 'title', 'similarity']

    most_similars = []
    for top_movie in top_movies:
        most_similars += get_most_similar_movies(movies_similarity, get_movie_name(movies, top_movie), get_movie_year(movies, top_movie))

    return pd.DataFrame(most_similars, columns=index).drop_duplicates().sort_values(by='similarity', 
                                                                                    ascending=False).head(nb_recommendations)

def get_collaborative_recommendations(dataframe,movies,users,users_similarity,user_id,rating_col='rating',user_id_col='user_id',
                        movie_id_col='movie_id',top=5,nb_recommendations=5):
   
    most_similar_users = get_most_similar_users(users_similarity,users,user_id,5)
    top_movies=[]
    for top_users in  most_similar_users:
        user_id=top_users[1]
        if(top_users != np.nan):
            movies_by_user=dataframe[dataframe[user_id_col] == user_id].sort_values(by=rating_col,
                                                                          ascending=False).head(top)[movie_id_col]
            for movie in movies_by_user:
                top_movies.append((movie,top_users[2]))
    
    index=['movie_id', 'title', 'similarity']

    most_similars = []
    for top_movie,similarity_value in top_movies:
        most_similars.append( (top_movie,get_movie_name(movies, top_movie), similarity_value))
    

    return pd.DataFrame(most_similars, columns=index).drop_duplicates().sort_values(by='similarity', 
                                                                                    ascending=True).head(nb_recommendations)
