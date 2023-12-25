import streamlit as st
import plotly.express as px

import pandas as pd
import pickle
from random import choice

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

with open('models//ohe.pickle', 'rb') as f:
    ohe_enc = pickle.load(f)

with open('models//scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

with open('models//model_regular.pickle', 'rb') as f:
    model_regular = pickle.load(f)

with open('models//model_tuned.pickle', 'rb') as f:
    model_tuned = pickle.load(f)

df = st.cache_data(pd.read_pickle)('data//df_clean.p')
y_test = pd.read_csv('data//y_test.csv')

prediction_regular = pd.read_csv('data//Predictions_regular_test.csv', index_col='ID')
prediction_tuned = pd.read_csv('data//Predictions_tuned_test.csv', index_col='ID')

models = {
    'cтандартная': {
        'type': 'regular',
        'name': 'логистическая регрессия',
        'params': 'полиномиальные признаки',
        'best_thr': 0.03,
    },
    'настроенная': {
        'type': 'tuned',
        'name': 'логистическая регрессия',
        'params': 'С=1.2, max_iter=1000',
        'best_thr': 0.11,
    },
}

charts_dict = {
    "Возраст": "AGE",
    "Размер кредита": "CREDIT",
    "Личный доход": "PERSONAL_INCOME",
    "Сфера деятельности": "GEN_INDUSTRY",
    "Профессия": "GEN_TITLE",
    "Регион проживания": "FACT_ADDRESS_PROVINCE",
}

charts_texts = {
    'Возраст':
        "Самая активная страта людей, пользующихся кредитами - от 25 до 40 лет. "
        "Потребность в кредитовании угасает к шестому десятку: "
        "после пятидесяти лет люди крайне редко прибегают к этой услуге. "
        "Это может быть связано как с устроенностью жизни людей к этому возрасту, так и с банальным эйджизмом: "
        "банки попросту отказывают в кредите людям этого возраста, связывая свои решения с большими рисками "
        "для заёмщиков данной категории в сфере трудоустройства и здоровья",
    'Размер кредита':
        'Судя по всему, банк занимается микрозаймами, так как суммы кредитов не превышают 92 тыс. руб.'
        'В целом, величины кредитов равномерно распределены вплоть до этой суммы. '
        'Однако больше всего берут кредиты на круглые суммы - 10 и 20 тысяч рублей.',
    'Личный доход':
        'Ежемесячный размер дохода клиентов крайне невелик и ограничивается в целом 52 тыс. руб.'
        'Это как раз характерно для обращающихся в микрокредитные организации людей.',
    'Сфера деятельности':
        'Больше всего берут кредит работники торговли, бюджетной сферы (здравохранение и гос.учреждения) и рабочих '
        'специальностей',
    'Профессия':
        'Как и предполагалось по результатам анализа сферы деятельности, за такими кредитами '
        'обращаются в основном люди рабочих специальностей: рабочие и специалисты',
    'Регион проживания':
        'Наибольшее число кредитов взяли люди, проживающие в Кемеровской и Читинской областях, а также Алтайском крае. '
        'Кроме того, много кредитов у людей, проживающих в Краснодарском крае. Предположу, что это связано с тем, что '
        'головная организация этого банка (а скорее всего, микрокредитной организации), '
        'расположена в Сибирском Федеральном округе, откуда она начала строить свою сеть в ЮФО'

}

pie_dict = {
    "Пол": "GENDER",
    "Образование": "EDUCATION",
    "Семейное положение": "MARITAL_STATUS",
    "Количество детей": "CHILD_TOTAL",
    "Наличие работы": "SOCSTATUS_WORK_FL",
    "Наличие пенсии": "SOCSTATUS_PENS_FL",
    "Обеспеченность жильём": "FL_PRESENCE_FL",
    "Количество авто": "OWN_AUTO",

}

pie_texts = {
    "Пол": "1 - мужчина, 0- женщина. Т.е. Мужчины занимают почти в 2 раза чаще женщин",
    "Образование": "Почти 72% занимающих имеют среднее образование. Однако каждый пятый из заёмщиков имеет высшее",
    "Семейное положение": "62% занимающих состоит в браке, 24 никогда не состояло. Меньше всего кредитов у разведенных",
    "Количество детей": "Берущие кредит имеют как правило до двух детей, иметь больше - большая редкость",
    "Наличие работы": "Кредиты дают только работающим :-)",
    "Наличие пенсии": "А пенсионерам почти не дают",
    "Обеспеченность жильём": "С наличием жилья у пользователей микрозаймов не очень",
    "Количество авто": "Что там квартира, машины у людей нет. Если имеешь две машины - за кредитом не идёшь.",

}

numeric_columns = [
    'AGREEMENT_RK',
    'ID',
    'TARGET',
    'SOCSTATUS_WORK_FL',
    'SOCSTATUS_PENS_FL',
    'FL_PRESENCE_FL',
    'OWN_AUTO',
    'GENDER'
]

target_distr_dict = {
    "Размер кредита": "CREDIT",
    "Личный доход": "PERSONAL_INCOME",
}

target_distr_texts = {
    "Размер кредита":
        "В целом, как было отмечено выше, особой зависимости целевой переменной от числовым признаков не обнаружено. "
        "Тем не менее, глядя на эту диаграмму рассеивания можно сказать, что держатели небольших сумм находятся в "
        "более зависимом положении и более склонны к сотрудничеству, чем крупные заёмщики",
    "Личный доход":
        "В целом, как было отмечено выше, особой зависимости целевой переменной от числовым признаков не "
        "обнаружено.Тем не менее, глядя на эту диаграмму рассеивания можно сказать, что больший личный доход "
        "позволяет людям быть равнодушным к предложениям банка",
}
