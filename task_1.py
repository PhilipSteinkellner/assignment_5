import nearest_neighbour_1 as nn
import pandas as pd

nn.split_data("./ml-1m/ratings.dat")

# UserID::MovieID::Rating::Timestamp
ratings_train = pd.read_csv('./ml-1m/ratings.dat_train', header=None, sep="::",
                      names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')
# ratings_train = ratings_train.sample(n=100, random_state=1)

ratings_test = pd.read_csv('./ml-1m/ratings.dat_test', header=None, sep="::",
                      names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')

# get all unique users
users = set(ratings_train['UserID'])

for user in users:
    similar_users = nn.get_similar_users(user, ratings_train) # retrieve dataframe with similar users

    # all movies rated by user
    movies_user = ratings_train[ratings_train['UserID'] == user][['MovieID', 'Rating']]

    # movies to predict a rating for
    movies_to_rate = ratings_test[ratings_test['UserID'] == user][['MovieID', 'Rating']]

    recommended_movies = nn.recommend_movies(user, similar_users, 11, ratings_train, movies_to_rate)
    print(recommended_movies)


