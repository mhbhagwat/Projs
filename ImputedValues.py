__author__ = 'mrunmayee'

import pandas as pd
import numpy as np
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn import neighbors
import matplotlib.patches as mpatches

np.random.seed(1)


# Function divide the data into training and test data sets. Input: Filename
def validate(da):
    """
    This function divides the input data set into training and testing subsets.
    :param da: The input data set.
    :return: Training and testing predictors and response variable.
    """
    seq = xrange(len(da))
    train1 = np.random.choice(seq, int(0.80 * len(da)), replace = False)
    test1 = set(seq).difference(set(train1))

    # Separate data into training and test sets
    tr_pred = np.array(da.loc[train1, 0:len(da.columns) - 2])
    tr_y = np.array(da.loc[train1, len(da.columns) - 1])
    test_pred = np.array(da.loc[test1, 0:len(da.columns) - 2])
    test_y = np.array(da.loc[test1, len(da.columns) - 1])
    return tr_pred, tr_y, test_pred, test_y


# Input: tr_pred_pima file
def introduce_missing_mean(tr_pred_pima, col, per):
    """
    This function inputs a training data set and inserts missing values into it. It uses mean method of imputation to
    replace the missing values.
    :param tr_pred_pima: Training data set in which missing values are inserted.
    :param col: List of attributes.
    :param per: List of percentage of missing values to be inserted.
    :return: Treated training data set.
    """
    data = pd.DataFrame(tr_pred_pima)
    seq = xrange(len(data))
    t1 = np.random.choice(seq, int(per * len(data)), replace = False)
    s1 = set(seq).difference(set(t1))

    # Introduce missing values
    for j in col:
        data.loc[t1, j] = np.NaN

        # Substitute missing values by mean
        mean_val = round(np.sum(np.array(data.loc[s1, j])) * 1.0 / len(s1), 3)
        data.loc[t1, j] = mean_val

    return data


def introduce_missing_knn(tr_pred, col, per):
    """
    This function inputs a training data set and inserts missing values into it. It uses k-NN (10-NN) method of
    imputation to replace the missing values.
    :param tr_pred_pima: Training data set in which missing values are inserted.
    :param col: List of attributes.
    :param per: List of percentage of missing values to be inserted.
    :return: Treated training data set.
    """
    data = pd.DataFrame(tr_pred)
    seq = xrange(len(data))
    t1 = np.random.choice(seq, int(per * len(data)), replace = False)
    s1 = set(seq).difference(set(t1))

    # Introduce missing values
    col_diff = list(set(list(data.columns.values)).difference(set(col)))

    for j in col:
        data.loc[t1, j] = np.NaN

        # Substitute missing values by mean
        knn = neighbors.KNeighborsRegressor(10)
        y = knn.fit(data.loc[s1, col_diff], data.loc[s1, j]).predict(data.loc[t1, col_diff])
        data.loc[t1, j] = y

    return data


def plot_pima(pima1, pima2, pima3, pimak1, pimak2, pimak3, per):
    """
    This function plots the graphs for Error Rate v/s percentage og missing values inserted for pima data sets.
    :return: None
    """

    # Legend
    blue_patch = mpatches.Patch(color='blue', label='Mean', linestyle = 'solid')
    green_patch = mpatches.Patch(color='green', label='10NN')

    plt.figure(1)
    plt.subplot(221)
    plt.axis([5, 65, 0.49, 0.65])
    plt.plot(per, pima1, 'bo', per, pima1, 'b--', per, pimak1, 'go', per, pimak1, 'g--')
    plt.xlabel('Percentage of examples with Missing Values', fontsize=10, color='black')
    plt.ylabel('Error Rate ', fontsize=10, color='black')
    plt.title('Missing Data Artificially Inserted into Attribute 1', fontsize=10)
    plt.legend(handles=[blue_patch, green_patch], loc='upper right', prop={'size':10})

    plt.subplot(222)
    plt.axis([5, 65, 0.55, 0.63])
    plt.plot(per, pima2, 'bo', per, pima2, 'b--', per, pimak2, 'go', per, pimak2, 'g--')
    plt.xlabel('Percentage of examples with Missing Values', fontsize=10, color='black')
    plt.ylabel('Error Rate ', fontsize=10, color='black')
    plt.title('Missing Data Artificially Inserted into Attribute 1, 5', fontsize=10)
    plt.legend(handles=[blue_patch, green_patch], loc='lower left', prop={'size':10})

    plt.subplot(223)
    plt.axis([5, 65, 0.54, 0.62])
    plt.plot(per, pima3, 'bo', per, pima3, 'b--', per, pimak3, 'go', per, pimak3, 'g--')
    plt.xlabel('Percentage of examples with Missing Values', fontsize=10, color='black')
    plt.ylabel('Error Rate ', fontsize=10, color='black')
    plt.title('Missing Data Artificially Inserted into Attribute 0, 1, 5', fontsize=10)

    plt.legend(handles=[blue_patch, green_patch], loc='upper center', prop={'size':10})

    plt.show()


def plot_bupa(bupa1, bupa2, bupa3, bupak1, bupak2, bupak3, per):
    """
    This function plots the graphs for Error Rate v/s percentage og missing values inserted for bupa data sets.
    :return: None
    """
    # Legend
    blue_patch = mpatches.Patch(color='blue', label='Mean', linestyle = 'solid')
    green_patch = mpatches.Patch(color='green', label='10NN')

    plt.figure(2)
    plt.subplot(221)
    plt.axis([5, 65, 0.52, 0.75])
    plt.plot(per, bupa1, 'bo', per, bupa1, 'b--', per, bupak1, 'go', per, bupak1, 'g--')
    plt.xlabel('Percentage of examples with Missing Values', fontsize=10, color='black')
    plt.ylabel('Error Rate ', fontsize=10, color='black')
    plt.title('Missing Data Artificially Inserted into Attribute 4', fontsize=10)
    plt.legend(handles=[blue_patch, green_patch], loc='lower right', prop={'size':10})

    plt.subplot(222)
    plt.axis([5, 65, 0.60, 0.80])
    plt.plot(per, bupa2, 'bo', per, bupa2, 'b--', per, bupak2, 'go', per, bupak2, 'g--')
    plt.xlabel('Percentage of examples with Missing Values', fontsize=10, color='black')
    plt.ylabel('Error Rate ', fontsize=10, color='black')
    plt.title('Missing Data Artificially Inserted into Attribute 2, 4', fontsize=10)
    plt.legend(handles=[blue_patch, green_patch], loc='upper right', prop={'size':10})

    plt.subplot(223)
    plt.axis([5, 65, 0.58, 0.75])
    plt.plot(per, bupa3, 'bo', per, bupa3, 'b--', per, bupak3, 'go', per, bupak3, 'g--')
    plt.xlabel('Percentage of examples with Missing Values', fontsize=10, color='black')
    plt.ylabel('Error Rate ', fontsize=10, color='black')
    plt.title('Missing Data Artificially Inserted into Attribute 2, 4, 5', fontsize=10)
    plt.legend(handles=[blue_patch, green_patch], loc='lower right', prop={'size':10})

    plt.show()


if __name__ == '__main__':
    bupa = pd.read_csv("bupa.txt", header = None)
    pima = pd.read_csv("pima.txt", header = None)

    per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    ############################
    # Pima data set
    ############################
    # Divide the data into training and test data set
    tr_pred_pima, tr_y_pima, test_pred_pima, test_y_pima = validate(pima)


    # Introduce and fill the missing values.
    # Mean/ mode
    pima1 = []
    pima2 = []
    pima3 = []
    for i in per:
        temp_train = tr_pred_pima
        temp_train = np.array(introduce_missing_mean(temp_train, [1], i))

        # Classification algorithm
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_pima)
        # RMSE
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_pima) - test_y_pima)) * 1.0 / len(test_pred_pima)))
        pima1.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_pima
        temp_train = np.array(introduce_missing_mean(temp_train, [1, 5], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_pima)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_pima) - test_y_pima)) * 1.0 / len(test_pred_pima)))
        pima2.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_pima
        temp_train = np.array(introduce_missing_mean(temp_train, [0, 1, 5], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_pima)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_pima) - test_y_pima)) * 1.0 / len(test_pred_pima)))
        pima3.append(round(rmse, 2))

    # KNN
    pimak1 = []
    pimak2 = []
    pimak3 = []
    for i in per:
        temp_train = tr_pred_pima
        temp_train = np.array(introduce_missing_knn(temp_train, [1], i))

        # Classification algorithm
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_pima)
        # RMSE
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_pima) - test_y_pima)) * 1.0 / len(test_pred_pima)))
        pimak1.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_pima
        temp_train = np.array(introduce_missing_mean(temp_train, [1, 5], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_pima)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_pima) - test_y_pima)) * 1.0 / len(test_pred_pima)))
        pimak2.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_pima
        temp_train = np.array(introduce_missing_mean(temp_train, [0, 1, 5], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_pima)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_pima) - test_y_pima)) * 1.0 / len(test_pred_pima)))
        pimak3.append(round(rmse, 2))


    ############################
    # Bupa data set
    ############################
    # Divide the data into training and test data set
    tr_pred_bupa, tr_y_bupa, test_pred_bupa, test_y_bupa = validate(bupa)

    # Introduce and fill the missing values.
    # Mean/ mode
    bupa1 = []
    bupa2 = []
    bupa3 = []
    for i in per:
        temp_train = tr_pred_bupa
        temp_train = np.array(introduce_missing_mean(temp_train, [4], i))

        # Classification algorithm
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_bupa)
        # RMSE
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_bupa) - test_y_bupa)) * 1.0 / len(test_pred_bupa)))
        bupa1.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_bupa
        temp_train = np.array(introduce_missing_mean(temp_train, [2, 4], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_bupa)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_bupa) - test_y_bupa)) * 1.0 / len(test_pred_bupa)))
        bupa2.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_bupa
        temp_train = np.array(introduce_missing_mean(temp_train, [2, 4, 5], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_bupa)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_bupa) - test_y_bupa)) * 1.0 / len(test_pred_bupa)))
        bupa3.append(round(rmse, 2))

    # KNN
    bupak1 = []
    bupak2 = []
    bupak3 = []
    for i in per:
        temp_train = tr_pred_bupa
        temp_train = np.array(introduce_missing_mean(temp_train, [4], i))

        # Classification algorithm
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_bupa)
        # RMSE
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_bupa) - test_y_bupa)) * 1.0 / len(test_pred_bupa)))
        bupak1.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_bupa
        temp_train = np.array(introduce_missing_mean(temp_train, [2, 4], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_bupa)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_bupa) - test_y_bupa)) * 1.0 / len(test_pred_bupa)))
        bupak2.append(round(rmse, 2))

    for i in per:
        temp_train = tr_pred_bupa
        temp_train = np.array(introduce_missing_mean(temp_train, [2, 4, 5], i))
        pree = ensemble.RandomForestClassifier()
        pree = pree.fit(temp_train, tr_y_bupa)
        rmse = np.sqrt((np.sum(np.square(pree.predict(test_pred_bupa) - test_y_bupa)) * 1.0 / len(test_pred_bupa)))
        bupak3.append(round(rmse, 2))

    # Plot data
    per1 = [10, 20, 30, 40, 50, 60]
    plot_pima(pima1, pima2, pima3, pimak1, pimak2, pimak3, per1)
    plot_bupa(bupa1, bupa2, bupa3, bupak1, bupak2, bupak3, per1)
