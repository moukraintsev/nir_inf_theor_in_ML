import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as la
import scipy
from scipy import integrate
from scipy.stats import gaussian_kde
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def count_my(number_of_rows, arr, species_, number_of_feature_):
    my = 0
    for j in range(number_of_rows):
        my += arr[j][number_of_feature_]
    my = my / number_of_rows
    print("For parameter ", number_of_feature_ + 1, '(', names_of_features[number_of_feature_], ') and species = ',
          species_)
    print("My:\t", my)
    return my


def count_sum(number_of_rows, arr, number_of_feature_, my):
    summa = 0
    for j in range(number_of_rows):
        summa += (arr[j][number_of_feature_] - my) ** 2
    summa = summa / number_of_rows
    print("Summa:\t", summa)
    return summa


def count_p(data_fragm, number_of_rows, number_of_feature_, my, summa):
    arr = np.zeros(shape=(number_of_rows, 4))
    arr_p = np.zeros(shape=(number_of_rows, 1))
    for j in range(number_of_rows):
        arr[j][number_of_feature_] = data_fragm.iloc[j][number_of_feature_]
    for j in range(number_of_rows):
        arr_p[j][0] = (math.exp(-0.5 * (arr[j][number_of_feature_] - my) ** 2 * summa)) / math.sqrt(2 * math.pi * summa)
    return arr_p


def build_graph_and_entropy(arr, data_fragm, species_, number_of_rows, number_of_feature, array_p, summa, my):
    entropy = 0
    for i in range(1, number_of_rows):
        entropy += array_p[i][0] * math.log2(1 / array_p[i][0])
    entropy *= -1
    print("Entropy SUMMOI\t= ", entropy, )
    names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # plt.plot(data_fragm[names[number_of_feature]][data_fragm['species'] == species_].tolist(), array_p, 'ro')

    spisok = np.zeros((1, 50))
    for i in range(0, number_of_rows):
        spisok[0][i] = arr[i][0]

    fig, ax = plt.subplots()
    ax.plot(data_fragm[names[number_of_feature]][data_fragm['species'] == species_].tolist(), array_p, 'ro')
    ax.grid()

    ax.set_xlabel('Вектор x')
    ax.set_ylabel('Плотность рас-ия p(x)')

    plt.title(names[number_of_feature] + ' ' + species_)
    plt.show()


def count_class(data_old, species_, number_of_feature_):
    fragm_data = data_old[data_old['species'] == species_]
    number_of_rows = len(fragm_data.index)
    arr = np.zeros(shape=(number_of_rows, 4))

    for j in range(number_of_rows):
        arr[j][number_of_feature_] = fragm_data.iloc[j][number_of_feature_]

    my = count_my(number_of_rows, arr, species_, number_of_feature_)
    summa = count_sum(number_of_rows, arr, number_of_feature_, my)
    arr_p = count_p(fragm_data, number_of_rows, number_of_feature_, my, summa)
    build_graph_and_entropy(arr, fragm_data, species_, number_of_rows, number_of_feature_, arr_p, summa, my)


# ====================================================
# многомерный случай
# ====================================================


def count_dataset_iris():
    # Считываем датасет, перегоняем в нумпай
    dataset = pd.read_csv('tableconvert_csv_crzwvc.csv')
    arr = dataset.to_numpy()

    # Инициализируем счетчики количества элементов в классах
    setosa_counter = 0
    versicolor_counter = 0
    virginica_counter = 0
    all_counter = 0

    # Инициализируем массивы с параметрами цветов по классам
    arr_setosa = np.array([])
    arr_versicolor = np.array([])
    arr_virginica = np.array([])

    # Разделение на классы с отбрасыванием названия
    for x in arr:
        if x[4] == 'setosa':
            if setosa_counter == 0:
                arr_setosa = np.hstack((arr_setosa, np.array(x)))
            else:
                arr_setosa = np.vstack((arr_setosa, np.array(x)))
            setosa_counter += 1
        elif x[4] == 'versicolor':
            if versicolor_counter == 0:
                arr_versicolor = np.hstack((arr_versicolor, np.array(x)))
            else:
                arr_versicolor = np.vstack((arr_versicolor, np.array(x)))
            versicolor_counter += 1
        elif x[4] == 'virginica':
            if virginica_counter == 0:
                arr_virginica = np.hstack((arr_virginica, np.array(x)))
            else:
                arr_virginica = np.vstack((arr_virginica, np.array(x)))
            virginica_counter += 1
        all_counter += 1

    arr_setosa = arr_setosa[::, 0:4]
    arr_versicolor = arr_versicolor[::, 0:4]
    arr_virginica = arr_virginica[::, 0:4]
    arr_all_data = arr[::, 0:4]

    # На этом этапе у нас 4 массива с параметрами (3 класса + 1 целый)
    # Если надо проверить раскоменитить это -->
    # print('setosa\n', arr_setosa, 'versicolor\n', arr_versicolor, 'virginica\n', arr_virginica, 'all\n', arr_all_data)

    # Инициализируем массивы для подсчета МЮ
    arr_my_setosa = np.array([0.0, 0.0, 0.0, 0.0])
    arr_my_versicolor = np.array([0.0, 0.0, 0.0, 0.0])
    arr_my_virginica = np.array([0.0, 0.0, 0.0, 0.0])
    arr_my_all_data = np.array([0.0, 0.0, 0.0, 0.0])

    # Считаем МЮ
    for i in range(4):
        for j in range(setosa_counter):
            arr_my_setosa[i] += arr_setosa[j][i]
    arr_my_setosa = arr_my_setosa / setosa_counter

    for i in range(4):
        for j in range(versicolor_counter):
            arr_my_versicolor[i] += arr_versicolor[j][i]
    arr_my_versicolor = arr_my_versicolor / versicolor_counter

    for i in range(4):
        for j in range(virginica_counter):
            arr_my_virginica[i] += arr_virginica[j][i]
    arr_my_virginica = arr_my_virginica / virginica_counter

    for i in range(4):
        for j in range(all_counter):
            arr_my_all_data[i] += arr_all_data[j][i]
    arr_my_all_data = arr_my_all_data / all_counter

    # На этом этапе у нас 4 массива с МЮ (3 класса + 1 целый)
    # Если надо проверить раскоменитить это и дописать один из массивов (arr_my_ВИД) -->
    # print(arr_my_all_data)

    # Считаем сумму
    arr_sum_setosa = arr_setosa
    arr_sum_versicolor = arr_versicolor
    arr_sum_virginica = arr_virginica
    arr_sum_all_data = arr_all_data

    for i in range(4):
        for j in range(setosa_counter):
            arr_sum_setosa[j][i] = arr_sum_setosa[j][i] - arr_my_setosa[i]

    for i in range(4):
        for j in range(versicolor_counter):
            arr_sum_versicolor[j][i] = arr_sum_versicolor[j][i] - arr_my_versicolor[i]

    for i in range(4):
        for j in range(virginica_counter):
            arr_sum_virginica[j][i] = arr_sum_virginica[j][i] - arr_my_virginica[i]

    for i in range(4):
        for j in range(all_counter):
            arr_sum_all_data[j][i] = arr_sum_all_data[j][i] - arr_my_all_data[i]

    # На этом этапе у нас 4 вспомогательных массива с суммой (3 класса + 1 целый)
    # Если надо проверить раскоменитить это и дописать один из массивов (arr_sum_ВИД) -->
    # print(arr_sum_all_data)

    # Транспонирование суммы
    arr_sum_setosa_t = arr_sum_setosa.transpose()
    arr_sum_versicolor_t = arr_sum_versicolor.transpose()
    arr_sum_virginica_t = arr_sum_virginica.transpose()
    arr_sum_all_data_t = arr_sum_all_data.transpose()

    # Финальный подсчет суммы
    arr_sum_multi_setosa = (np.dot(arr_sum_setosa_t, arr_sum_setosa) / setosa_counter).astype('float')
    arr_sum_multi_versicolor = (np.dot(arr_sum_versicolor_t, arr_sum_versicolor) / versicolor_counter).astype('float')
    arr_sum_multi_virginica = (np.dot(arr_sum_virginica_t, arr_sum_virginica) / virginica_counter).astype('float')
    arr_sum_multi_all_data = (np.dot(arr_sum_all_data_t, arr_sum_all_data) / all_counter).astype('float')

    # На этом этапе у нас 4  массива с суммой (3 класса + 1 целый)
    # Если надо проверить раскоменитить это и дописать один из массивов (arr_sum_multi_ВИД) -->
    # print(arr_sum_multi_setosa)

    # Здесь считаем плотность
    arr_p_setosa = np.zeros(shape=(setosa_counter, 1))
    arr_p_versicolor = np.zeros(shape=(setosa_counter, 1))
    arr_p_virginica = np.zeros(shape=(setosa_counter, 1))

    for i in range(setosa_counter):
        arr_p_setosa[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_sum_setosa[i], la.inv(arr_sum_multi_setosa)), arr_sum_setosa[i])) / math.sqrt(
            (2 * math.pi) ** 4 * la.det(arr_sum_multi_setosa))

    for i in range(versicolor_counter):
        arr_p_versicolor[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_sum_versicolor[i], la.inv(arr_sum_multi_versicolor)),
                            arr_sum_versicolor[i])) / math.sqrt(
            (2 * math.pi) ** 4 * la.det(arr_sum_multi_versicolor))

    for i in range(virginica_counter):
        arr_p_virginica[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_sum_virginica[i], la.inv(arr_sum_multi_virginica)),
                            arr_sum_virginica[i])) / math.sqrt(
            (2 * math.pi) ** 4 * la.det(arr_sum_multi_virginica))

    # Считаем энтропию
    h_setosa = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 4 * la.det(arr_sum_multi_setosa))
    h_versicolor = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 4 * la.det(arr_sum_multi_versicolor))
    h_virginica = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 4 * la.det(arr_sum_multi_virginica))
    h_all = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 4 * la.det(arr_sum_multi_all_data))

    print(h_setosa, h_versicolor, h_virginica, h_all)


def count_dataset_vine():
    # Считываем датасет, перегоняем в нумпай
    dataset = pd.read_csv('winequality-red.csv')
    arr = dataset.to_numpy()

    parameters_list = list(dataset)

    # Инициализируем массивы с параметрами вин по классам
    arr_class_3 = np.array([])
    arr_class_4 = np.array([])
    arr_class_5 = np.array([])
    arr_class_6 = np.array([])
    arr_class_7 = np.array([])
    arr_class_8 = np.array([])

    # Инициализируем счетчики количества элементов в классах
    class_3_counter = 0
    class_4_counter = 0
    class_5_counter = 0
    class_6_counter = 0
    class_7_counter = 0
    class_8_counter = 0
    class_all_counter = 0

    # Разделение на классы с отбрасыванием названия
    for x in arr:
        if x[11] == 3:
            if class_3_counter == 0:
                arr_class_3 = np.hstack((arr_class_3, np.array(x)))
            else:
                arr_class_3 = np.vstack((arr_class_3, np.array(x)))
            class_3_counter += 1
        elif x[11] == 4:
            if class_4_counter == 0:
                arr_class_4 = np.hstack((arr_class_4, np.array(x)))
            else:
                arr_class_4 = np.vstack((arr_class_4, np.array(x)))
            class_4_counter += 1
        elif x[11] == 5:
            if class_5_counter == 0:
                arr_class_5 = np.hstack((arr_class_5, np.array(x)))
            else:
                arr_class_5 = np.vstack((arr_class_5, np.array(x)))
            class_5_counter += 1
        elif x[11] == 6:
            if class_6_counter == 0:
                arr_class_6 = np.hstack((arr_class_6, np.array(x)))
            else:
                arr_class_6 = np.vstack((arr_class_6, np.array(x)))
            class_6_counter += 1
        elif x[11] == 7:
            if class_7_counter == 0:
                arr_class_7 = np.hstack((arr_class_7, np.array(x)))
            else:
                arr_class_7 = np.vstack((arr_class_7, np.array(x)))
            class_7_counter += 1
        elif x[11] == 8:
            if class_8_counter == 0:
                arr_class_8 = np.hstack((arr_class_8, np.array(x)))
            else:
                arr_class_8 = np.vstack((arr_class_8, np.array(x)))
            class_8_counter += 1
        class_all_counter += 1

    arr_class_3 = arr_class_3[::, 0:11]
    arr_class_4 = arr_class_4[::, 0:11]
    arr_class_5 = arr_class_5[::, 0:11]
    arr_class_6 = arr_class_6[::, 0:11]
    arr_class_7 = arr_class_7[::, 0:11]
    arr_class_8 = arr_class_8[::, 0:11]
    arr_class_all = arr[::, 0:11]

    # Проверка корректности разделения
    if (class_3_counter + class_4_counter + class_5_counter + class_6_counter + class_7_counter + class_8_counter
            != class_all_counter):
        print("Error in separation")

    # Инициализируем массивы для подсчета МЮ
    arr_class_3_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arr_class_4_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arr_class_5_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arr_class_6_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arr_class_7_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arr_class_8_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arr_class_all_my = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Считаем МЮ
    for i in range(11):
        for j in range(class_3_counter):
            arr_class_3_my[i] += arr_class_3[j][i]
    arr_class_3_my = arr_class_3_my / class_3_counter

    for i in range(11):
        for j in range(class_4_counter):
            arr_class_4_my[i] += arr_class_4[j][i]
    arr_class_4_my = arr_class_4_my / class_4_counter

    for i in range(11):
        for j in range(class_5_counter):
            arr_class_5_my[i] += arr_class_5[j][i]
    arr_class_5_my = arr_class_5_my / class_5_counter

    for i in range(11):
        for j in range(class_6_counter):
            arr_class_6_my[i] += arr_class_6[j][i]
    arr_class_6_my = arr_class_6_my / class_6_counter

    for i in range(11):
        for j in range(class_7_counter):
            arr_class_7_my[i] += arr_class_7[j][i]
    arr_class_7_my = arr_class_7_my / class_7_counter

    for i in range(11):
        for j in range(class_8_counter):
            arr_class_8_my[i] += arr_class_8[j][i]
    arr_class_8_my = arr_class_8_my / class_8_counter

    for i in range(11):
        for j in range(class_all_counter):
            arr_class_all_my[i] += arr_class_all[j][i]
    arr_class_all_my = arr_class_all_my / class_all_counter

    # Считаем сумму
    arr_class_3_sum = np.copy(arr_class_3)
    arr_class_4_sum = np.copy(arr_class_4)
    arr_class_5_sum = np.copy(arr_class_5)
    arr_class_6_sum = np.copy(arr_class_6)
    arr_class_7_sum = np.copy(arr_class_7)
    arr_class_8_sum = np.copy(arr_class_8)
    arr_class_all_sum = np.copy(arr_class_all)

    for i in range(11):
        for j in range(class_all_counter):
            arr_class_all_sum[j][i] = arr_class_all_sum[j][i] - arr_class_all_my[i]

    for i in range(11):
        for j in range(class_3_counter):
            arr_class_3_sum[j][i] = arr_class_3_sum[j][i] - arr_class_3_my[i]

    for i in range(11):
        for j in range(class_4_counter):
            arr_class_4_sum[j][i] = arr_class_4_sum[j][i] - arr_class_4_my[i]

    for i in range(11):
        for j in range(class_5_counter):
            arr_class_5_sum[j][i] = arr_class_5_sum[j][i] - arr_class_5_my[i]

    for i in range(11):
        for j in range(class_6_counter):
            arr_class_6_sum[j][i] = arr_class_6_sum[j][i] - arr_class_6_my[i]

    for i in range(11):
        for j in range(class_7_counter):
            arr_class_7_sum[j][i] = arr_class_7_sum[j][i] - arr_class_7_my[i]

    for i in range(11):
        for j in range(class_8_counter):
            arr_class_8_sum[j][i] = arr_class_8_sum[j][i] - arr_class_8_my[i]

    # Транспонируем сумму
    arr_class_3_sum_t = arr_class_3_sum.transpose()
    arr_class_4_sum_t = arr_class_4_sum.transpose()
    arr_class_5_sum_t = arr_class_5_sum.transpose()
    arr_class_6_sum_t = arr_class_6_sum.transpose()
    arr_class_7_sum_t = arr_class_7_sum.transpose()
    arr_class_8_sum_t = arr_class_8_sum.transpose()
    arr_class_all_sum_t = arr_class_all_sum.transpose()

    # Сумма итоговая
    arr_class_3_sum_multi = (np.dot(arr_class_3_sum_t, arr_class_3_sum) / class_3_counter).astype('float')
    arr_class_4_sum_multi = (np.dot(arr_class_4_sum_t, arr_class_4_sum) / class_4_counter).astype('float')
    arr_class_5_sum_multi = (np.dot(arr_class_5_sum_t, arr_class_5_sum) / class_5_counter).astype('float')
    arr_class_6_sum_multi = (np.dot(arr_class_6_sum_t, arr_class_6_sum) / class_6_counter).astype('float')
    arr_class_7_sum_multi = (np.dot(arr_class_7_sum_t, arr_class_7_sum) / class_7_counter).astype('float')
    arr_class_8_sum_multi = (np.dot(arr_class_8_sum_t, arr_class_8_sum) / class_8_counter).astype('float')
    arr_class_all_sum_multi = (np.dot(arr_class_all_sum_t, arr_class_all_sum) / class_all_counter).astype('float')

    # Инициализируем массивы под плотность
    arr_class_3_p = np.zeros(shape=(class_3_counter, 1))
    arr_class_4_p = np.zeros(shape=(class_4_counter, 1))
    arr_class_5_p = np.zeros(shape=(class_5_counter, 1))
    arr_class_6_p = np.zeros(shape=(class_6_counter, 1))
    arr_class_7_p = np.zeros(shape=(class_7_counter, 1))
    arr_class_8_p = np.zeros(shape=(class_8_counter, 1))

    # Считаем плонтность
    for i in range(class_3_counter):
        arr_class_3_p[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_class_3_sum[i], la.inv(arr_class_3_sum_multi)), arr_class_3_sum[i])) / math.sqrt(
            (2 * math.pi) ** 11 * la.det(arr_class_3_sum_multi))

    for i in range(class_4_counter):
        arr_class_4_p[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_class_4_sum[i], la.inv(arr_class_4_sum_multi)), arr_class_4_sum[i])) / math.sqrt(
            (2 * math.pi) ** 11 * la.det(arr_class_4_sum_multi))

    for i in range(class_5_counter):
        arr_class_5_p[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_class_5_sum[i], la.inv(arr_class_5_sum_multi)), arr_class_5_sum[i])) / math.sqrt(
            (2 * math.pi) ** 11 * la.det(arr_class_5_sum_multi))

    for i in range(class_6_counter):
        arr_class_6_p[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_class_6_sum[i], la.inv(arr_class_6_sum_multi)), arr_class_6_sum[i])) / math.sqrt(
            (2 * math.pi) ** 11 * la.det(arr_class_6_sum_multi))

    for i in range(class_7_counter):
        arr_class_7_p[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_class_7_sum[i], la.inv(arr_class_7_sum_multi)), arr_class_7_sum[i])) / math.sqrt(
            (2 * math.pi) ** 11 * la.det(arr_class_7_sum_multi))

    for i in range(class_8_counter):
        arr_class_8_p[i] = math.exp(
            -0.5 * np.inner(np.dot(arr_class_8_sum[i], la.inv(arr_class_8_sum_multi)), arr_class_8_sum[i])) / math.sqrt(
            (2 * math.pi) ** 11 * la.det(arr_class_8_sum_multi))

    # Считаем энтропию
    class_3_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_3_sum_multi))
    class_4_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_4_sum_multi))
    class_5_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_5_sum_multi))
    class_6_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_6_sum_multi))
    class_7_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_7_sum_multi))
    class_8_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_8_sum_multi))
    class_all_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_all_sum_multi))

    print(class_3_h, class_4_h, class_5_h, class_6_h, class_7_h, class_8_h, class_all_h)
    print('=============================')

    # Вводим коррекцию для положительной энтропии
    def change(array):
        tau = 0.00015075
        d = tau + np.diag(array)
        for ik in range(11):
            for jk in range(11):
                if ik == jk:
                    array[ik][jk] = d[ik]
        return array

    # Считаем сумму с коррекцией
    arr_class_3_sum_multi = change(arr_class_3_sum_multi)
    arr_class_4_sum_multi = change(arr_class_4_sum_multi)
    arr_class_5_sum_multi = change(arr_class_5_sum_multi)
    arr_class_6_sum_multi = change(arr_class_6_sum_multi)
    arr_class_7_sum_multi = change(arr_class_7_sum_multi)
    arr_class_8_sum_multi = change(arr_class_8_sum_multi)
    arr_class_all_sum_multi = change(arr_class_all_sum_multi)

    # Считаем энтропию с коррекцией
    class_3_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_3_sum_multi))
    class_4_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_4_sum_multi))
    class_5_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_5_sum_multi))
    class_6_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_6_sum_multi))
    class_7_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_7_sum_multi))
    class_8_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_8_sum_multi))
    class_all_h = 0.5 * math.log((2 * math.pi * math.exp(1)) ** 11 * la.det(arr_class_all_sum_multi))

    class_3_h_norm = class_3_h / class_3_counter
    class_4_h_norm = class_4_h / class_4_counter
    class_5_h_norm = class_5_h / class_5_counter
    class_6_h_norm = class_6_h / class_6_counter
    class_7_h_norm = class_7_h / class_7_counter
    class_8_h_norm = class_8_h / class_8_counter
    class_all_h_norm = class_all_h / class_all_counter

    print("Энтропия:\t", class_3_h, class_4_h, class_5_h, class_6_h, class_7_h, class_8_h, class_all_h)
    print("Энтропия norm:\t", class_3_h_norm, class_4_h_norm, class_5_h_norm, class_6_h_norm, class_7_h_norm,
          class_8_h_norm, class_all_h_norm)

    def entropy_class_graph():
        x = [3, 4, 5, 6, 7, 8]
        y = [class_3_h, class_4_h, class_5_h, class_6_h, class_7_h, class_8_h]
        y_n = [class_3_h_norm, class_4_h_norm, class_5_h_norm, class_6_h_norm, class_7_h_norm, class_8_h_norm]

        fig, ax = plt.subplots()
        plt.title("График зависимости энтропии от метки класса")
        ax.plot(x, y)
        ax.grid()
        ax.set_xlabel('Метка класса')
        ax.set_ylabel('Энтропия')
        plt.show()

        fig, ax = plt.subplots()
        plt.title("График зависимости энтропии от метки класса")
        ax.plot(x, y_n)
        ax.grid()
        ax.set_xlabel('Метка класса')
        ax.set_ylabel('Энтропия нормализованная')
        plt.show()

    # entropy_class_graph()

    def prepare_graph(array, form, is_by_class):
        # Готовит данные для построения графика
        # На вход получает:
        # array - массив со значением Recall и Precision
        # form - 0, если Recall; 1, если Precision;
        # is_by_class - 0, если не для меток класса; 1 - для меток
        # Возвращает списки - оси X (энтропия) и Y (значения Recall или Precision)

        # Инициализирует списки
        y_list = []
        x_list_entr = []
        x_list_entr_norm = []
        x_list_ = []
        iteration = 0
        if is_by_class == 1:
            for ig in array[form]:
                y_list.append(ig)
            return x_list_entr, y_list, x_list_entr_norm
        for ig in array[form]:
            if ig != 0:
                y_list.append(ig)
                x_list_.append(iteration + 3)
            iteration += 1
        for ig in x_list_:
            if ig == 3:
                x_list_entr.append(class_3_h)
                x_list_entr_norm.append(class_3_h_norm)
            elif ig == 4:
                x_list_entr.append(class_4_h)
                x_list_entr_norm.append(class_4_h_norm)
            elif ig == 5:
                x_list_entr.append(class_5_h)
                x_list_entr_norm.append(class_5_h_norm)
            elif ig == 6:
                x_list_entr.append(class_6_h)
                x_list_entr_norm.append(class_6_h_norm)
            elif ig == 7:
                x_list_entr.append(class_7_h)
                x_list_entr_norm.append(class_7_h_norm)
            elif ig == 8:
                x_list_entr.append(class_8_h)
                x_list_entr_norm.append(class_8_h_norm)
        return x_list_entr, y_list, x_list_entr_norm

    def build_graph(array, prediction_method):
        xy_recall = prepare_graph(array, 0, 0)
        xy_precision = prepare_graph(array, 1, 0)
        xy_recall_class = prepare_graph(array, 0, 1)
        xy_precision_class = prepare_graph(array, 1, 1)
        # Строим графики

        # Обычная энтропия
        fig, ax = plt.subplots()
        plt.title("Способ предсказания " + prediction_method)
        ax.plot(xy_recall[0], xy_recall[1])
        ax.grid()
        ax.set_xlabel('Энтропия')
        ax.set_ylabel('Recall')
        plt.show()

        fig, ax = plt.subplots()
        plt.title("Способ предсказания " + prediction_method)
        ax.plot(xy_precision[0], xy_precision[1])
        ax.grid()

        ax.set_xlabel('Энтропия')
        ax.set_ylabel('Precision')
        plt.show()

        # Нормализованная энтропия
        fig, ax = plt.subplots()
        plt.title("Способ предсказания " + prediction_method)
        ax.plot(xy_recall[2], xy_recall[1])
        ax.grid()
        ax.set_xlabel('Энтропия нормализованная')
        ax.set_ylabel('Recall')
        plt.show()

        fig, ax = plt.subplots()
        plt.title("Способ предсказания " + prediction_method)
        ax.plot(xy_precision[2], xy_precision[1])
        ax.grid()

        ax.set_xlabel('Энтропия нормализованная')
        ax.set_ylabel('Precision')
        plt.show()

        # Метка класса
        fig, ax = plt.subplots()
        plt.title("Способ предсказания " + prediction_method)
        x_line = [3, 4, 5, 6, 7, 8]
        print('==========================================')
        print(xy_recall[0])
        ax.plot(x_line, xy_recall_class[1], 'red', label = 'Recall')
        ax.plot(x_line, xy_precision_class[1], 'darkmagenta', label = 'Precision')
        ax.grid()
        ax.legend()
        ax.set_xlabel('Метка класса')
        plt.show()

        # Энтропия Precision + Recall
        fig, ax = plt.subplots()
        plt.title("Способ предсказания " + prediction_method)
        x_line = xy_recall[2]
        x_line_ = xy_precision[2]
        print()
        ax.plot(x_line, xy_recall[1], 'red', label='Recall')
        ax.plot(x_line_, xy_precision[1], 'darkmagenta', label='Precision')
        ax.grid()
        ax.legend()
        ax.set_xlabel('Энтропия нормализованная')
        plt.show()

    # Получаем массив значений Recall и Precision
    array_1 = predict_1()
    array_2 = predict_2()
    array_3 = predict_3()

    build_graph(array_1, 'SGDClassifier')
    build_graph(array_2, 'LogisticRegression')
    build_graph(array_3, 'make_pipeline')

    def build_graph_p(data_fragm, length, parameter, array_p, n_class):
        x_vector = []
        y_vector = []
        for element in range(0, length):
            x_vector.append(data_fragm[element][parameter])
        for element in range(0, length):
            y_vector.append(round(array_p[element][0] / 1e15).astype('int'))

        data = []
        for i in range(0, length):
            data += [x_vector[i]] * y_vector[i]

        density = gaussian_kde(data)
        xs = np.linspace(0, 7, 200)
        density.covariance_factor = lambda: .25
        density._compute_covariance()

        fig, ax = plt.subplots()
        plt.plot(xs, density(xs))
        ax.set_xlabel('Вектор x')
        ax.set_ylabel('Плотность рас-ия p(x)')
        ax.grid()

        plt.title(parameters_list[parameter] + ' class ' + str(n_class))
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(x_vector, y_vector, 'ro')
        ax.grid()

        ax.set_xlabel('Вектор x')
        ax.set_ylabel('Плотность рас-ия p(x)')

        plt.title(parameters_list[parameter] + ' class ' + str(n_class))
        plt.show()

    # build_graph_p(arr_class_3, class_3_counter, 3, arr_class_3_p, 3)


def predict_1():
    with open('winequality-red.csv') as f:
        X = csv.reader(f)
        Wine = list(X)
    for i in Wine:
        del Wine[0]
    Wine_train, Wine_test = train_test_split(Wine, test_size=0.3, random_state=42)
    Classes_train = [i.pop() for i in Wine_train]
    Classes_test = [i.pop() for i in Wine_test]
    print(Classes_test)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000)
    clf.fit(Wine_train, Classes_train)

    Wine_test_ = []
    j = 0
    for elem in Wine_test:
        Wine_test_.append(list(map(float, Wine_test[j])))
        j += 1

    Classes_pred = clf.predict(Wine_test_)
    # print('Предсказание:')
    # print(Classes_pred)
    print('Recall:')
    print(recall_score(Classes_test, Classes_pred, average=None))
    print('Precision:')
    print(precision_score(Classes_test, Classes_pred, average=None))
    return recall_score(Classes_test, Classes_pred, average=None), precision_score(Classes_test, Classes_pred,
                                                                                   average=None)


def predict_2():
    with open('winequality-red.csv') as f:
        X = csv.reader(f)
        Wine = list(X)
    for i in Wine:
        del Wine[0]
    Wine_train, Wine_test = train_test_split(Wine, test_size=0.3, random_state=42)
    Classes_train = [i.pop() for i in Wine_train]
    Classes_test = [i.pop() for i in Wine_test]
    # print(Classes_test)

    clf2 = LogisticRegression(random_state=0).fit(Wine_train, Classes_train)
    clf2.fit(Wine_train, Classes_train)

    Wine_test_ = []
    j = 0
    for elem in Wine_test:
        Wine_test_.append(list(map(float, Wine_test[j])))
        j += 1

    Classes_pred_LR = clf2.predict(Wine_test_)

    return recall_score(Classes_test, Classes_pred_LR, average=None), precision_score(Classes_test, Classes_pred_LR,
                                                                                      average=None)


def predict_3():
    with open('winequality-red.csv') as f:
        X = csv.reader(f)
        Wine = list(X)
    for i in Wine:
        del Wine[0]
    Wine_train, Wine_test = train_test_split(Wine, test_size=0.3, random_state=42)
    Classes_train = [i.pop() for i in Wine_train]
    Classes_test = [i.pop() for i in Wine_test]
    print(Classes_test)

    clf3 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf3.fit(Wine_test, Classes_test)

    Wine_test_ = []
    j = 0
    for elem in Wine_test:
        Wine_test_.append(list(map(float, Wine_test[j])))
        j += 1

    Classes_pred_SVC = clf3.predict(Wine_test_)

    return recall_score(Classes_test, Classes_pred_SVC, average=None), precision_score(Classes_test, Classes_pred_SVC,
                                                                                       average=None)


data = pd.read_csv('tableconvert_csv_crzwvc.csv')

species = ['setosa', 'versicolor', 'virginica']
names_of_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


# count_dataset_iris()

# count_class(data, 'setosa', 0)
# count_class(data, 'setosa', 1)
# count_class(data, 'setosa', 2)
# count_class(data, 'setosa', 3)
#
# count_class(data, 'versicolor', 0)
# count_class(data, 'versicolor', 1)
# count_class(data, 'versicolor', 2)
# count_class(data, 'versicolor', 3)
#
# count_class(data, 'virginica', 0)
# count_class(data, 'virginica', 1)
# count_class(data, 'virginica', 2)
# count_class(data, 'virginica', 3)

count_dataset_vine()
data_new = pd.read_csv('winequality-red.csv')
