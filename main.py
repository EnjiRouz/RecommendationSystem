'''
Система рекомендаций - это программное обеспечение, которое анализирует доступные данные,
чтобы сделать предложения чего-то, что может заинтересовать пользователя.

Существует в основном четыре типа рекомендательных механизмов:
    1. Механизм рекомендаций на основе контента;
    2. Механизм рекомендаций на основе коллаборативной фильтрации;
    3. Механизм рекомендаций на основе популярности;
    4.Гибридная система рекомендаций;

В данной системе рекомендаций использован механизм рекомендаций на основе контента.
Система принимает проект, который нравится пользователю, а затем анализирует его,
чтобы получить список похожих проектов (например, по ключевым словам), используя оценку сходства.

Ресурсы:
https://medium.com/code-heroku/building-a-project-recommendation-engine-in-python-using-scikit-learn-c7489d7cb145
'''

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# комбинирование всех значимых параметров в одну строку
def combine_features(row):
    return row["keywords"] + " " + row["programming_language"] + " " + row["roles"]


# вспомогательная функция для получения названия проекта по индексу
def get_title_from_index(index):
    return data_frame[data_frame.index == index]["title"].values[0]


# вспомогательная функция для получения индекса по названию проекта
def get_index_from_title(title):
    return data_frame[data_frame.title == title]["index"].values[0]


if __name__ == '__main__':

    # загрузка датасета с проектами
    data_frame = pd.read_csv("dataset.csv")

    # столбцы, по которым будет осущствляться поиск
    features = ["keywords", "programming_language", "roles"]

    # если существуют незаполненные поля - заполняем их при помощи пустой строки (для csv-формата)
    for feature in features:
        data_frame[feature] = data_frame[feature].fillna('')

    # создание отдельного столбца с комбинированными параметрами
    data_frame["combined_features"] = data_frame.apply(combine_features, axis=1)

    # преобразование набора текста в матрицу (вектора) подсчетов токенов (маркеров)
    count_matrix = CountVectorizer().fit_transform(data_frame["combined_features"])
    # print(count_matrix.toarray())

    # получение косинусной матрицы подобия
    cosine_sim = cosine_similarity(count_matrix)
    # print(cosine_sim)

    # получение интересуемого проекта (допустим, пользователь нажал на определённый проект =>
    # внизу можно ему показать топ-3 похожих)
    project_user_likes = "АНАЛИЗ ТЕКСТОВ ВАКАНСИЙ"

    # получеие индекса интересуемого проекта
    project_index = get_index_from_title(project_user_likes)

    # Доступ к строке через индекс проектов, соответствующий этому проекту (понравившемуся проекту) в матрице подобия
    # нужен, чтобы получить оценки подобия всех других проектов из текущего проекта

    # Далее происходит перечисление всех оценок сходства этого проекта,
    # чтобы создать кортеж из индекса проекта и оценок сходства.

    # Ряд оценок сходства преобразуется из этого [5 0.6 0.3 0.9] в это [(0, 5) (1, 0.6) (2, 0.3) (3, 0.9)]
    # При этом каждый элемент списка помещается в эту форму (индекс проекта, оценка сходства)
    similar_projects = list(enumerate(cosine_sim[project_index]))

    # Сортировка списка похожих проектов по степени сходства в порядке убывания
    # Поскольку самый похожий проект - это он сам, мы отбросим первый элемент после сортировки
    sorted_similar_projects = sorted(similar_projects, key=lambda x: x[1], reverse=True)[1:]

    # Вывод отсортированных похожих проектов для проекта, который нравится пользователю
    # Кортежи имеют вид (индекс проекта, значение подобия)
    # print(sorted_similar_projects)

    # Вывод первых N записей из отсортированного списка похожих проектов и оценки сходства
    i = 0
    print("Топ 2 похожих проектов на " + project_user_likes + ":")
    for i in range(len(sorted_similar_projects)):
        print(" *", get_title_from_index(sorted_similar_projects[i][0]),
              " Процент подобия: ", "%.2f" % (sorted_similar_projects[i][1] * 100))
        i = i + 1
        if i >= 2:
            break
