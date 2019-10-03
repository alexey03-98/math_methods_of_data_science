"""
Implementations of util functions for all tasks

:author: Aleksei Neliubin
"""

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns


def plot_graph_with_predictions_by_methods(experiment_data, target_data, test_size,
                                           method_for_prediction, x_label, y_label, plot_title):
    output_file = open('accuracies.txt', "w")

    train_features_data, test_features_data, train_target_data, test_target_data = \
        train_test_split(experiment_data, target_data, test_size=test_size, random_state=42)
    method_for_prediction.fit(train_features_data, train_target_data)

    x_min, x_max = test_features_data[:, 0].min() - .5, test_features_data[:, 0].max() + .5
    y_min, y_max = test_features_data[:, 1].min() - .5, test_features_data[:, 1].max() + .5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    prediction = method_for_prediction.predict(np.c_[xx.ravel(), yy.ravel()])
    prediction_for_accuracy = method_for_prediction.predict(test_features_data)
    prediction = prediction.reshape(xx.shape)

    output_file.write('\nAccuracy for graph ' + plot_title + ' for features [' + x_label + '_' + y_label + ']\n')
    output_file.write(str(accuracy_score(test_target_data, prediction_for_accuracy)))

    plt.figure()

    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.pcolormesh(xx, yy, prediction, cmap=plt.cm.Paired)
    plt.scatter(test_features_data[:, 0], test_features_data[:, 1], c=test_target_data,
                edgecolors='k', cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())


def draw_predicted_by_linear_discriminant_data_set(experiment_data):
    linear_discriminant = LinearDiscriminantAnalysis()
    linear_discriminant.fit(experiment_data.data, experiment_data.target)
    predicted_data_set = linear_discriminant.predict(experiment_data.data)
    predicted_data_frame = DataFrame(experiment_data.data)
    predicted_data_frame.columns = experiment_data.feature_names
    predicted_data_frame['target'] = predicted_data_set
    predicted_data_frame['name'] = __name_column(predicted_data_set)
    sns.pairplot(predicted_data_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
        , 'name']], hue='name')
    wrong_precision_data, wrong_precision_target_predicted, wrong_precision_target_corrected =\
        __get_wrong_predicted_points(experiment_data.data, experiment_data.target, predicted_data_set)
    wrong_precision_data = __unite_vectors(wrong_precision_data, wrong_precision_data)
    wrong_precision_target = __unite_vectors(wrong_precision_target_predicted, wrong_precision_target_corrected)
    wrong_precision_data_frame = DataFrame(wrong_precision_data)
    wrong_precision_data_frame.columns = experiment_data.feature_names
    wrong_precision_data_frame['target'] = wrong_precision_target
    wrong_precision_data_frame['name'] = __name_column(wrong_precision_target)
    sns.pairplot(wrong_precision_data_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                             'petal width (cm)', 'name']], hue='name')


def __name_column(target_data):
    result = []
    size = len(target_data)
    for i in range(size):
        if target_data[i] == 0:
            result.append('setosa')
        if target_data[i] == 1:
            result.append('versicolor')
        if target_data[i] == 2:
            result.append('virginica')
        if target_data[i] == 3:
            result.append('setosa_predicted')
        if target_data[i] == 4:
            result.append('versicolor_predicted')
        if target_data[i] == 5:
            result.append('virginica_predicted')
    return result


def __get_wrong_predicted_points(experiment_data, experiment_target, predicted_data):
    result_data = []
    result_target_predicted = []
    result_target_corrected = []
    size = len(experiment_data)
    for i in range(size):
        if experiment_target[i] != predicted_data[i]:
            result_data.append(experiment_data[i])
            result_target_predicted.append(predicted_data[i] + 3)
            result_target_corrected.append(experiment_target[i])
    return result_data, result_target_predicted, result_target_corrected


def __unite_vectors(vector1, vector2):
    result = []
    size1 = len(vector1)
    size2 = len(vector2)
    for i in range(size1):
        result.append(vector1[i])
    for i in range(size2):
        result.append(vector2[i])
    return result
