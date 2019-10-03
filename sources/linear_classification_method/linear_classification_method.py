from sklearn import datasets
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
from sources.util import plot_graph_with_predictions_by_methods
from sources.util import draw_predicted_by_linear_discriminant_data_set
from sources.util import calculate_functions


def main():
    # _______________________________________________First part of task_________________________________________________
    output_file = open('output.txt', "w")

    iris_data = datasets.load_iris()
    test_size = 0.15

    iris_frame = DataFrame(iris_data.data)
    iris_frame.columns = iris_data.feature_names
    iris_frame['target'] = iris_data.target
    iris_frame['name'] = iris_frame.target.apply(lambda x: iris_data.target_names[x])

    feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

    correlation = iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].corr()
    output_file.write("Correlation for all table\n\n")
    output_file.write(correlation.to_string())

    output_file.write("\n\nCorrelation for data grouped by class name\n\n")

    group_correlation = \
        iris_frame[['name', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].\
            groupby('name').corr()
    output_file.write(group_correlation.to_string())
    output_file.close()

    sns.pairplot(iris_frame[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'name']],
                 hue='name')
    # __________________________________________________________________________________________________________________

    # ____________________________________________Second part of task___________________________________________________
    # ______________________________________________Common objects______________________________________________________

    first_two_features = iris_data.data[:, :2]
    target = iris_data.target

    # __________________________________________________________________________________________________________________

    # __________________________________________Linear discriminant_____________________________________________________

    linear_discriminant = LinearDiscriminantAnalysis()

    plot_graph_with_predictions_by_methods(first_two_features, target, test_size, linear_discriminant,
                                           feature_names[0], feature_names[1], 'Linear discriminant')

    # __________________________________________________________________________________________________________________

    # __________________________________________Quadratic discriminant__________________________________________________

    quadratic_discriminant = QuadraticDiscriminantAnalysis()

    plot_graph_with_predictions_by_methods(first_two_features, target, test_size, quadratic_discriminant,
                                           feature_names[0], feature_names[1], 'Quadratic discriminant')

    # __________________________________________________________________________________________________________________

    # ________________________________________Logistic Regression_______________________________________________________

    logistic_regression = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

    plot_graph_with_predictions_by_methods(first_two_features, target, test_size, logistic_regression,
                                           feature_names[0], feature_names[1], 'Logistic Regression')

    # _________________________________________________SVM method_______________________________________________________
    # _________________________________________SVM method with linear kernel____________________________________________

    svm_method_with_linear_kernel = SVC(kernel='linear')

    plot_graph_with_predictions_by_methods(first_two_features, target, test_size, svm_method_with_linear_kernel,
                                           feature_names[0], feature_names[1], 'SVM method with linear kernel')

    # __________________________________________________________________________________________________________________

    # _________________________________________SVM method with quadratic kernel_________________________________________

    svm_method_with_quadratic_kernel = SVC()

    plot_graph_with_predictions_by_methods(first_two_features, target, test_size, svm_method_with_quadratic_kernel,
                                           feature_names[0], feature_names[1], 'SVM method with quadratic kernel')

    # __________________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________________

    # ______________________________________________Third part of task__________________________________________________

    draw_predicted_by_linear_discriminant_data_set(iris_data)

    # ______________________________________________Fourth part of task_________________________________________________

    calculate_functions(first_two_features, target)

    # __________________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________________
    plt.show()


if __name__ == '__main__':
    main()