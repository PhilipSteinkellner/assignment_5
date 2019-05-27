
import pandas as pd
import math
import nearest_neighbour_1 as nn



def evaluate_pres_recall():

    trainfile = './ml-1m/ratings_train.dat'
    testfile = './ml-1m/ratings_test.dat'


    # traindata
    ratings_train = pd.read_csv(trainfile, header=None, sep="::",
                                names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')
    # ratings = ratings.sample(n=10000, random_state=1)

    # testdata
    ratings_test = pd.read_csv(testfile, header=None, sep="::",
                               names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python')

    # get all unique users
    users_train = set(ratings_train['UserID'])
    users_test = set(ratings_test['UserID'])

    movies = pd.read_csv('./ml-1m/movies.dat', header=None, encoding="ISO-8859-1", sep="::",
                         names=["MovieID", "Title", "Genres"], engine='python')

    #declaration of the ground truth table
    tp = 0
    fp = 0
    fn = 0
    tn = 0



    for user in users_train:
        print("working on: ", user)

        # get similiar users
        similar_users = pd.Series()
        similar_users = nn.get_similar_users(user, ratings_train)

        movies_to_rate = ratings_test[ratings_test['UserID'] == user][['MovieID', 'Rating']]

        # recommend movies to the specified user
        # amount of ratings of users the algorithm considers for the computation
        neighborhood_size = 3
        recommended_movies = nn.recommend_movies(user, similar_users ,neighborhood_size, ratings_train ,movies_to_rate)


        # relevant has to be  >3


        for ind, m in recommended_movies.head(10).iteritems():
            #print(ind)
            #print(m)

            for index, row in movies_to_rate.iterrows():
                # print(row['Rating'])
                if ind == row['MovieID']:
                    #print(ind)

                    #true positive
                    if row['Rating'] > 3 and m > 3:
                        tp = tp + 1

                    #false positive
                    elif m > 3 and row['Rating'] <=3:
                        fp = fp +1

                    #false negative
                    elif m <= 3 and row['Rating'] >3:
                        fn = fn +1

    print(tp)
    print(fp)
    print(fn)

    print('Preciscion: ', tp/(tp+fp))
    print('Recall: ', tp/(tp+fn))





if __name__ == '__main__':

    evaluate_pres_recall()



