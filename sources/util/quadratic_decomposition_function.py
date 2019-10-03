"""
File with implementations of needless for decomposition method functions

:author: Aleksei Neliubin
"""

from sympy import *


def calculate_functions(experiment_data, target):
    first_class_data = __decompose_data_by_classes(experiment_data, target, 0)
    second_class_data = __decompose_data_by_classes(experiment_data, target, 1)
    third_class_data = __decompose_data_by_classes(experiment_data, target, 2)
    x, y = symbols('x y')
    x_min = min(experiment_data[:, 0])
    x_max = max(experiment_data[:, 0])
    y_min = min(experiment_data[:, 1])
    y_max = max(experiment_data[:, 1])

    probabilities = [
        __calculate_probability(target, 0),
        __calculate_probability(target, 1),
        __calculate_probability(target, 2)
    ]

    mathematical_expectations = [
        __calculate_mathematical_expectation(first_class_data),
        __calculate_mathematical_expectation(second_class_data),
        __calculate_mathematical_expectation(third_class_data)
    ]
    covariance_matrices = [
        __calculate_covariance_matrix(first_class_data, mathematical_expectations[0]),
        __calculate_covariance_matrix(second_class_data, mathematical_expectations[1]),
        __calculate_covariance_matrix(third_class_data, mathematical_expectations[2])
    ]
    first_function = Eq(simplify(__build_function_equation([covariance_matrices[0], covariance_matrices[1]],
                                                           [mathematical_expectations[0], mathematical_expectations[1]],
                                                           [probabilities[0], probabilities[1]], x, y)))
    second_function = Eq(simplify(__build_function_equation([covariance_matrices[1], covariance_matrices[2]],
                                                            [mathematical_expectations[1],
                                                             mathematical_expectations[2]],
                                                            [probabilities[1], probabilities[2]], x, y)))
    third_function = Eq(simplify(__build_function_equation([covariance_matrices[0], covariance_matrices[2]],
                                                           [mathematical_expectations[0], mathematical_expectations[2]],
                                                           [probabilities[0], probabilities[2]], x, y)))

    graph1 = plot_implicit(first_function, (x, x_min, x_max), (y, y_min, y_max), show=False, line_color='blue')
    graph2 = plot_implicit(second_function, (x, x_min, x_max), (y, y_min, y_max), show=False, line_color='red')
    graph3 = plot_implicit(third_function, (x, x_min, x_max), (y, y_min, y_max), show=False, line_color='green')

    graph1.append(graph2[0])
    graph1.append(graph3[0])
    graph1.show()


# Information about mining of q1 and q2 locate on lecture mmadCapter1Bayes
def __build_function_equation(covariance_matrices, mathematical_expectations, probabilities, x, y):
    variables_minus_mean = Matrix([x - mathematical_expectations[0][0], y - mathematical_expectations[0][1]])
    q1 = Matrix(covariance_matrices[0])
    det_q1 = q1.det()
    q1 = q1.inv()
    q1 = variables_minus_mean.transpose() * q1 * variables_minus_mean
    variables_minus_mean = Matrix([x - mathematical_expectations[1][0], y - mathematical_expectations[1][1]])
    q2 = Matrix(covariance_matrices[1])
    det_q2 = q2.det()
    q2 = q2.inv()
    q2 = variables_minus_mean.transpose() * q2 * variables_minus_mean
    return sympify((q2 - q1)[0, 0] + ln(sqrt(det_q2) / sqrt(det_q1)) + ln(probabilities[0] / probabilities[1]))


def __decompose_data_by_classes(experiment_data, experiment_data_target, class_index):
    experiment_data_size = len(experiment_data)
    class_data = []
    for i in range(experiment_data_size):
        if experiment_data_target[i] == class_index:
            class_data.append([experiment_data[i][0], experiment_data[i][1]])
    return class_data


def __calculate_probability(experiment_data_target, class_index):
    n = len(experiment_data_target)
    n1 = 0

    for i in range(n):
        if experiment_data_target[i] == class_index:
            n1 = n1 + 1

    return n1 / n


def __calculate_mathematical_expectation(experiment_data):
    mathematical_expectations = []
    experiment_data_size = len(experiment_data)
    first_variable_sum = 0
    second_variable_sum = 0
    for i in range(experiment_data_size):
        first_variable_sum = first_variable_sum + experiment_data[i][0]
        second_variable_sum = second_variable_sum + experiment_data[i][1]

    mathematical_expectations.append(first_variable_sum / experiment_data_size)
    mathematical_expectations.append(second_variable_sum / experiment_data_size)
    return mathematical_expectations


def __calculate_dispersion(experiment_data, mathematical_expectations):
    dispersions = []
    experiment_data_size = len(experiment_data)
    first_variable_sum = 0
    second_variable_sum = 0
    for i in range(experiment_data_size):
        first_variable_sum = first_variable_sum + (experiment_data[i][0] - mathematical_expectations[0]) ** 2
        second_variable_sum = second_variable_sum + (experiment_data[i][1] - mathematical_expectations[1]) ** 2

    dispersions.append(first_variable_sum / (experiment_data_size - 1))
    dispersions.append(second_variable_sum / (experiment_data_size - 1))
    return dispersions


def __calculate_covariance_matrix(experiment_data, mathematical_expectations):
    covariance_matrix = []
    non_diagonal_element = 0
    diagonal_elements = __calculate_dispersion(experiment_data, mathematical_expectations)
    experiment_data_size = len(experiment_data)
    for i in range(experiment_data_size):
        non_diagonal_element = non_diagonal_element + (experiment_data[i][0] - mathematical_expectations[0]) * \
                               (experiment_data[i][1] - mathematical_expectations[1])
    non_diagonal_element = non_diagonal_element / (experiment_data_size - 1)
    covariance_matrix.append([diagonal_elements[0], non_diagonal_element])
    covariance_matrix.append([non_diagonal_element, diagonal_elements[1]])
    return covariance_matrix
