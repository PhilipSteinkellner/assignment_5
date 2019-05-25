import pandas as pd
import math
import traceback
import os
import random

# pandas option to display all columns when printing
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1080)


# prints up to 15 movies the specified user has rated
# parameters: user_a -> UserID of the user
def show_movies(user_a):

    try:
        # UserID::MovieID::Rating::Timestamp
        ratings = pd.read_csv('./ml-1m/ratings.dat', header=None, sep="::",
                              names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')
        # MovieID::Title::Genres
        movies = pd.read_csv('./ml-1m/movies.dat', header=None, encoding="ISO-8859-1", sep="::",
                             names=["MovieID", "Title", "Genres"], engine='python')

        # merge dataframes rating and movies
        movies_ratings = movies.merge(ratings, how='inner', on="MovieID")

        # select all movies rated by the specified user
        user_ratings = movies_ratings.loc[movies_ratings['UserID'] == user_a]

        # print these movies
        print(user_ratings.iloc[0:10][['Title', 'Genres','Rating']])
    except Exception as e:
        print('Something went wrong')
        print(e)
        traceback.print_exc()


# creates a pandas.Series object, index is the UserID of the different users, value is the similarity those users have
# in connection with the specified user_a
# parameters: user_a -> UserID of the user for which the similarity values should be generated
# the Series object will be serialized and stored as similar_users.pkl
def get_similar_users(user_a):

    try:
        # check if a similar_users file is already present for the specified user, if not generate one
        if os.path.isfile('./similar_users_' + str(user_a) + '.pkl'):
            # read the file
            similar_users = pd.read_pickle('./similar_users_' + str(user_a) + '.pkl')
            return similar_users

        # UserID::MovieID::Rating::Timestamp
        ratings = pd.read_csv('./ml-1m/ratings.dat', header=None, sep="::",
                              names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')
        # ratings = ratings.sample(n=10000, random_state=1)

        # get all unique users
        users = set(ratings['UserID'])
        # remove user that will receive recommendations
        users.remove(user_a)

        # all movies rated by user_a
        movies_user_a = ratings[ratings['UserID'] == user_a][['MovieID', 'Rating']]

        # series to store similarity values for other users
        similar_users = pd.Series()

        for user_b in users:

            # all movies rated by user_b
            movies_user_b = ratings[ratings['UserID'] == user_b][['MovieID', 'Rating']]

            # movies in common between user_a and user_b
            movies_in_common = set(movies_user_b['MovieID']).intersection(set(movies_user_a['MovieID']))

            if len(movies_in_common) == 0:          # no movies in common
                similar_users[str(user_b)] = -1
            else:
                # relevant movies user_a
                relevant_movies_a = movies_user_a[movies_user_a['MovieID'].isin(movies_in_common)]
                # relevant movies user_b
                relevant_movies_b = movies_user_b[movies_user_b['MovieID'].isin(movies_in_common)]

                # mean ratings
                mean_rating_user_a = relevant_movies_a['Rating'].mean()
                mean_rating_user_b = relevant_movies_b['Rating'].mean()

                # formula to calculate the similarity between the users (Pearson correlation)
                # check slides (VC Recommender Systems P01 Intro and CF -> slide 30)
                x = y1 = y2 = 0
                for index, row in relevant_movies_a.iterrows():

                    rating_a = row['Rating']
                    rating_b = relevant_movies_b[relevant_movies_b['MovieID'] == row['MovieID']][['Rating']].iloc[0]['Rating']

                    x = x + (rating_a - mean_rating_user_a) * (rating_b - mean_rating_user_b)
                    y1 = y1 + pow((rating_a - mean_rating_user_a), 2)
                    y2 = y2 + pow((rating_b - mean_rating_user_b), 2)

                # calculate similarity between user_a and user_b
                if x == 0:
                    similarity = 0
                else:
                    similarity = x / (math.sqrt(y1) * math.sqrt(y2))

                # add similarity value for user_b to the series
                similar_users[str(user_b)] = similarity

        # sort user by their similarity values in descending order
        similar_users = similar_users.sort_values(ascending=False)

        #print(similar_users)
        # serialize series object and save it
        similar_users.to_pickle('./similar_users_' + str(user_a) + '.pkl')

        return similar_users
    except Exception as e:
        print('Something went wrong')
        print(e)
        traceback.print_exc()


# recommends up to 10 movies for a specific user that the user has not yet seen
# parameters: user_a -> UserID of the user for which the movie recommendations should be generated
#             similar_users -> pandas.Series object which contains the users with their similarity values in descending order
#             neighborhood_size -> amount of ratings of users the algorithm considers for the computation
def recommend_movies(user_a, similar_users, neighborhood_size):
    try:
        # check if a recommended_movies file is already present for the specified user, if not generate one
        if os.path.isfile('./recommended_movies_' + str(user_a) + '.pkl'):
            # read the file
            movies_to_recommend = pd.read_pickle('./recommended_movies_' + str(user_a) + '.pkl')
        else:
            movies_to_recommend = pd.Series()

            # UserID::MovieID::Rating::Timestamp
            ratings = pd.read_csv('./ml-1m/ratings.dat', header=None, sep="::",
                                  names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')

            user_item_table = ratings.pivot(index="UserID", columns="MovieID", values="Rating")

            for movie, rating in user_item_table.loc[user_a].iteritems():
                # only rate the first 100 movies because of computational time
                if movie == 100:
                    break
                if math.isnan(rating):  # movie not seen by the user
                    movie_predicted_rating = 0
                    amount_of_ratings = 0

                    for user, value in similar_users.iteritems():
                        # get the users rating
                        rating_similar_user = user_item_table.loc[int(user)].loc[movie]
                        # check if similar user has rated the movie
                        if not math.isnan(rating_similar_user):
                            movie_predicted_rating += rating_similar_user
                            amount_of_ratings += 1

                        # if sufficient ratings have been considered exit the for loop
                        if amount_of_ratings >= neighborhood_size:
                            break

                    # calculate average of the ratings
                    movie_predicted_rating /= neighborhood_size

                    # fill the blank spot in the table with the generated estimated rating
                    user_item_table.loc[user_a].loc[movie] = movie_predicted_rating
                    movies_to_recommend.loc[movie] = movie_predicted_rating

            # sort movie list with movies with highest recommendations score on top
            movies_to_recommend = movies_to_recommend.sort_values(ascending=False)
            movies_to_recommend.to_pickle('./recommended_movies_' + str(user_a) + '.pkl')

        # MovieID::Title::Genres
        movies = pd.read_csv('./ml-1m/movies.dat', header=None, encoding="ISO-8859-1", sep="::",
                             names=["MovieID", "Title", "Genres"], engine='python')

        # iterate over the first 10 elements and print their data
        for index, value in movies_to_recommend.head(10).iteritems():
            print(movies.loc[movies['MovieID'] == index].to_string(index=False), "rating: ", value)

    except Exception as e:
        print('Something went wrong')
        traceback.print_exc()
        print(e)


def split_data (filename):

    with open(filename, "r") as f:
        data = f.read().split('\n')

    random.shuffle(data)

    train_data = data[:int((len(data) + 1) * .80)]  # Remaining 80% to training set
    test_data = data[int(len(data) * .80 + 1):]  # Splits 20% data to test set

    print(test_data)


    #write test
    with open(filename + '_test', 'w') as f:
        for item in test_data:
            f.write("%s\n" % item)

    #write training
    with open(filename + '_train', 'w') as f:
        for item in train_data:
            f.write("%s\n" % item)






if __name__ == '__main__':

    # Exercise A
    user_id = int(input("Enter user_id:"))

    split_data("./ml-1m/movies.dat")


    # Exercise B
    # show_movies(user_id)

    # Exercise C

    # C.1: first get user with high similarity score
    similar_users = get_similar_users(user_id)

    # C.2: recommend movies to the specified user
    # amount of ratings of users the algorithm considers for the computation
    neighborhood_size = 11
    recommend_movies(user_id, similar_users, neighborhood_size)

