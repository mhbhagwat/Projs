__author__ = 'mrunmayee'

import pandas as pd
import numpy as np
import argparse


def parseArgument():
    """
    :return: The function returns the arguments given on the command line. In this case the names of the
    training and testing files.
    """

    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = (parser.parse_args())
    return args.train[0], args.test[0]


def file_read(trainfile, testfile):
    """
    This function takes the training and testing files as input and applies collaborative filtering algorithm
    on the training file data and tests the results on the testing file. It prints the mean absolute error and
    the root mean square error of the test file. It also writes the predicted ratings for the testing file with the
    along with the testing file data to a file "predictions.txt"
    :param trainfile: Name of the training file.
    :param testfile: Name of the testing file.
    :return: None
    """

    train = pd.read_csv(trainfile, header = None)
    test = pd.read_csv(testfile, header = None)

    dict_users = {}                                                     # {User: {Movie: Rating}}
    dict_movies = {}
    for m in train.itertuples():
        if m[2] not in dict_users:
            dict_users[m[2]] = {}
            dict_users[m[2]][m[1]] = m[3]
        else:
            dict_users[m[2]][m[1]] = m[3]
        if m[1] not in dict_movies:
            dict_movies[m[1]] = {}
            dict_movies[m[1]][m[2]] = m[3]
        else:
            dict_movies[m[1]][m[2]] = m[3]

    user_avg_rating = {}                                            # Average movie rating per user
    dict_corr = {}                                                  # Correlation dictionary
    rating_diff = sqr_diff = 0
    num = den = d2 = 0
    list_ratings = []

    # Read test data set
    old = 0
    wf = open("predictions.txt", 'w')
    for first in test.itertuples():
        if first[1]: moviek = first[1]
        if first[2]: userk = first[2]


        # Implementation of the Cold start problem
        # New user, old movie
        if userk not in dict_users and moviek in dict_movies:
            print "New user"
            nu_userlist = dict_movies[moviek].keys()
            cs_total = 0
            for r in nu_userlist:
                cs_total += dict_movies[moviek][r]
            nu_rating = cs_total * 1.0 / len(nu_userlist)
            # print moviek, userk, nu_rating


        # New movie, old user
        elif moviek not in dict_movies and userk in dict_users:
            print "New movie"
            if userk not in user_avg_rating:
                user_avg_rating[userk] = round(np.mean(dict_users[userk].values()), 3)
            nm_rating = user_avg_rating[userk]


        # New user and new movie
        elif userk not in dict_users and moviek not in dict_movies:
            print "New user and new movie"
            nb_total = 0
            for i in dict_users.keys():
                for j in dict_users[i].keys():
                    nb_total += dict_users[i][j]

            nmu_rating = nb_total * 1.0 / len(train)


        else:
            old = 1
            if first[3]: actual = first[3]                              # Actual prediction

            user_moviek = dict_movies[moviek].keys()                    # Users who have rated the given movie


            # Average user rating
            if userk not in user_avg_rating:
                user_avg_rating[userk] = round(np.mean(dict_users[userk].values()), 3)
            for k in user_moviek:
                if k not in user_avg_rating:
                    user_avg_rating[k] = round(np.mean(dict_users[k].values()), 3)

            # Part 1 - Calculating correlation

            s1 = set(dict_users[userk].keys())

            for j in user_moviek:
                s2 = set(dict_users[j].keys())
                common = list(s1.intersection(s2))
                if len(common) != 0:
                    num = den1 = den2 = 0
                    for i in common:                        # j is a user, i is a movie common to j and userk
                        b1 = dict_users[userk][i] - user_avg_rating[userk]
                        b2 = dict_users[j][i] - user_avg_rating[j]
                        num += b1 * b2
                        den1 += np.square(b1)
                        den2 += np.square(b2)
                    if num == 0:
                        w = 0
                    else:
                        w = num * 1.0 / np.sqrt(den1 * den2)
                else:
                    w = 0
                s = tuple(sorted([userk, j]))
                if s not in dict_corr:
                    dict_corr[s] = round(w, 3)

        # Part 2 - Prediction
            for p in user_moviek:
                d1 = dict_users[p][moviek] - user_avg_rating[p]
                s1 = tuple(sorted([userk, p]))
                d2 = dict_corr[s1]
                num += d1 * d2
                den += abs(d2)
            if num == 0:
                rating = user_avg_rating[userk]
            else:
                rating = user_avg_rating[userk] + (num * 1.0 / den)

            rating_diff += abs(rating - actual)
            sqr_diff += np.square(rating - actual)
            list_ratings.append(round(rating, 2))

            # Write to 'predictions.txt' file
            wf.write("%s,%s,%s,%s\n" %(first[1], first[2], first[3], round(rating, 3)))

    if old == 1:                 # Flag to check if all the input cases do not belong to the cold start problem
        mean_abs_error = rating_diff * 1.0 / len(test)
        rmse = np.sqrt(sqr_diff * 1.0/ len(test))
        print "Mean absolute error:", round(mean_abs_error, 3)
        print "Root mean squared error", round(rmse, 3)
    wf.close()


if __name__ == '__main__':

    (trainfile, testfile) = parseArgument()
    file_read(trainfile, testfile)
