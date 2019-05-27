import nearest_neighbour_1 as nn
import pandas as pd
import math


def evaluate_mae_rmse():

    # split the data 80% trainings data and 20% test data
    nn.split_data("./ml-1m/ratings.dat")

    # Training Data
    ratings_train = pd.read_csv('./ml-1m/ratings_train.dat', header=None, sep="::",
                          names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')

    # Test Data
    ratings_test = pd.read_csv('./ml-1m/ratings_test.dat', header=None, sep="::",
                          names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')

    # get all unique users
    users = set(ratings_train['UserID'])

    # Neighborhood Size for the algorithm
    neighborhood_size = 11
    print("Neighborhood Size: ", neighborhood_size)

    # all generated predictions
    all_predictions = pd.DataFrame(columns=['UserID', 'MovieID', 'Rating'])

    # for all users make movie recommendation
    for user in users:
        print("Making Movie Recommendations for User", user)
        similar_users = nn.get_similar_users(user, ratings_train) # retrieve dataframe with similar users

        # all movies rated by user
        movies_user = ratings_train[ratings_train['UserID'] == user][['MovieID', 'Rating']]

        # movies to predict a rating for
        movies_to_rate = ratings_test[ratings_test['UserID'] == user][['MovieID', 'Rating']]

        # get predicted rating for the movies
        recommended_movies = nn.recommend_movies(user, similar_users, neighborhood_size, ratings_train, movies_to_rate)

        # add all the predictive ratings for the user to the all_predictions dataframe object
        list = []
        for key, value in recommended_movies.iteritems():
            dict = {
                'UserID': user,
                'MovieID': key,
                'Rating': value
            }
            list.append(dict)
        df = pd.DataFrame(list)
        all_predictions = all_predictions.append(df, sort=True)

    # Calculate MAE and RMSE
    amount_of_values = sum_mae = sum_rmse = 0

    # for all recommendations that have been made
    for index, row in all_predictions.iterrows():
        # predicted rating by the algorithm
        predicted_rating = float(row['Rating'])

        ratings_test2 = ratings_test[ratings_test['UserID'] == row['UserID']]
        ratings_test2 = ratings_test2[ratings_test2['MovieID'] == row['MovieID']]
        # actual rating made by the user
        actual_rating = float(ratings_test2['Rating'])

        diff = predicted_rating - actual_rating
        amount_of_values += 1

        # calculations for mae
        diff_mae = abs(diff)
        sum_mae += diff_mae

        # calculations for rmse
        diff_rmse = pow(diff, 2)
        sum_rmse += diff_rmse

    # Mean Absolute Error
    mae = sum_mae / amount_of_values

    # Root Mean Square Error
    rmse = math.sqrt(sum_rmse / amount_of_values)

    print("MAE: ", mae)
    print("RMSE: ", rmse)

    # Considering predicted ratings for 10 users:

    # MAE for Neighborhood Size 7: 0.9406156604901375
    # RMSE for Neighborhood Size 7: 1.1606775146034336

    # MAE for Neighborhood Size 11: 0.8648000544996256
    # RMSE for Neighborhood Size 11: 1.0833402930239178

    # MAE for Neighborhood Size 17: 0.8692156813636133
    # RMSE for Neighborhood Size 17: 1.1087648084272397


if __name__ == '__main__':

    evaluate_mae_rmse()
