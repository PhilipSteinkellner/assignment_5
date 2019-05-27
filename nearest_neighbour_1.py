import pandas as pd
import math
import traceback
import random

# creates a pandas.Series object, index is the UserID of the different users, value is the similarity those users have
# in connection with the specified user_a
# parameters: user_a -> UserID of the user for which the similarity values should be generated
# the Series object will be serialized and stored as similar_users.pkl
def get_similar_users(user_a, ratings):

    try:

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

        return similar_users

    except Exception as e:
        print('Something went wrong')
        print(e)
        traceback.print_exc()


# recommends up to 10 movies for a specific user that the user has not yet seen
# parameters: user_a -> UserID of the user for which the movie recommendations should be generated
#             similar_users -> pandas.Series object which contains the users with their similarity values in descending order
#             neighborhood_size -> amount of ratings of users the algorithm considers for the computation
def recommend_movies(user_a, similar_users, neighborhood_size, ratings, movies_to_rate):
    try:
        movies_to_recommend = pd.Series()

        for key, row in movies_to_rate.iterrows():

            movie_predicted_rating = 0
            amount_of_ratings = 0

            for user, value in similar_users.iteritems():
                # get the movies ratings
                rating_similar_user = ratings[ratings['MovieID'] == row['MovieID']]
                # get the users ratings
                rating_similar_user = rating_similar_user[rating_similar_user['UserID'] == int(user)]

                rating_similar_user = rating_similar_user['Rating']
                # check if similar user has rated the movie
                if not rating_similar_user.empty:
                    movie_predicted_rating += int(rating_similar_user)
                    amount_of_ratings += 1

                # if sufficient ratings have been considered exit the for loop
                if amount_of_ratings >= neighborhood_size:
                    break

            # if no predictive rating could be generated due to missing data
            if amount_of_ratings == 0:
                movie_predicted_rating = 0
            else:
                # calculate average of the ratings
                movie_predicted_rating /= amount_of_ratings

            # add the predicted movie rating to the series
            movies_to_recommend.loc[row['MovieID']] = movie_predicted_rating

        # sort movie list with movies with highest recommendations score on top
        movies_to_recommend = movies_to_recommend.sort_values(ascending=False)
        return movies_to_recommend

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

    print(train_data)

    #write test
    with open(filename + '_test', 'w') as f:
        for item in test_data:
            f.write("%s\n" % item)

    #write training
    with open(filename + '_train', 'w') as f:
        for item in train_data:
            f.write("%s\n" % item)