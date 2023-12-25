"""
Module responsible for content and display of the "Прогнозы" tab in streamlit application
Модуль, отвечающий за содержание и отображение вкладки "Прогнозы" в streamlit-приложении
"""

from imports import st, pd, choice
from imports import prediction_tuned, prediction_regular, y_test, models, df, ohe_enc, scaler, model_regular, \
    model_tuned
from imports import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_metrics_score(y_test, y_pred) -> dict:
    """
    Computes quality metrics for the passed true class values and their predictions, and stores them in a dictionary
    Считает метрики качества для переданных значений истинных классов и их прогнозов, и сохраняет их в словарь

    :param y_test: pandas.series containing test values of targets
    :param y_pred: pandas.series containing predictions  of targets
    :return: a dictionary of quality metrics
    """
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def get_metrics_score_thr(threshold: float, best_thr: float, model_type: str = 'regular') -> tuple[dict, tuple]:
    """
    Computes a set of metrics for the optimal and user thresholds, returns dictionaries of the computed metrics
    Вычисляет набор метрик для оптимального и пользовательского порогов, возвращает словари вычисленных метрик
    """
    pred_probability = prediction_regular if model_type == 'regular' else prediction_tuned
    y_thr = pred_probability > threshold
    y_best_thr = pred_probability > best_thr
    metrics = {'user': get_metrics_score(y_test, y_thr), 'best': get_metrics_score(y_test, y_best_thr)}
    y_thr_tuple = (y_thr, y_best_thr)
    return metrics, y_thr_tuple


def get_metrics(threshold: float, best_thr: float, model_type: str = 'regular') -> None:
    """
    Calculates differences of metrics for optimal and custom thresholds, and plots the resulting ratios using special features of the streamlit library
    Вычисляет разницы метрик для оптимального и пользовательского порогов, и рисует полученные соотношения с помощью специальных возможностей библиотеки streamlit
    """
    metrics, _ = get_metrics_score_thr(threshold=threshold, best_thr=best_thr, model_type=model_type)
    col1, col2, col3, col4 = st.columns(4)
    mets_name = list(metrics['user'].keys())
    pairs = zip(mets_name, [col1, col2, col3, col4])

    for mets, col in pairs:
        col.metric(mets, metrics['user'][mets], round((metrics['user'][mets] - metrics['best'][mets]), 4))


def get_single_prediction(single_df: pd.DataFrame, threshhold: float, best_thr: float, model_type) -> tuple[
    float, bool, bool]:
    """
    Applies trained classification models to a single user's data, organized as a dataframe, and returns a
    prediction. Depending on the model type, different preprocessing of the dataset is performed.

    Применяет обученные модели классификации к данным одного пользователя, оформленных в виде датафрейма, и возвращает
    прогноз. В зависимости от типа модели производится различная предобработка датасета.
    """
    if model_type == 'regular':
        predict_model = model_regular
        single_pred_positive = predict_model.predict_proba(single_df)[:, 1][0]
    else:
        predict_model = model_tuned
        single_pred_positive = predict_model.predict_proba(single_df)[:, 1][0]
    is_recommend_thr = single_pred_positive >= threshhold
    is_recommend_best_thr = single_pred_positive >= best_thr

    return single_pred_positive, is_recommend_thr, is_recommend_best_thr


def transform_df_regular_to_tuned(single_df_regular: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the frame to the required format for the piplane of the customized logistic regression model
    Приводит фрейм к необходимому формату для пайплайна настроенной модели логистической регрессии
    """
    single_df_tuned = pd.DataFrame(ohe_enc.transform(single_df_regular))
    single_df_tuned = pd.DataFrame(scaler.transform(single_df_tuned), columns=single_df_tuned.columns,
                                   index=single_df_tuned.index)
    return single_df_tuned


def collect_dataframe():
    """
    Provides input of user questionnaire data to build a customized forecast and turns it into a dataframe
    Обеспечивает ввод пользовательских данных для получения индивидуального прогноза и превращает их в датафрейм
    """
    with st.expander('Разверни кат для ввода анкеты'):
        st.subheader('Общие сведения о клиенте:')
        GENDER = st.radio("Пол", ("Мужской", "Женский"), horizontal=True)
        AGE = st.slider("Возраст", min_value=1, max_value=80, value=39, step=1)
        FACT_ADDRESS_PROVINCE = st.selectbox("Регион проживания", tuple(df.FACT_ADDRESS_PROVINCE.unique().tolist()))
        EDUCATION = st.radio("Образование", tuple(df.EDUCATION.unique().tolist()), horizontal=True)
        SOCSTATUS_WORK_FL = st.radio("Наличие работы", ("Есть", "Нет"), horizontal=True)
        SOCSTATUS_PENS_FL = st.radio("Наличие пенсии", ("Есть", "Нет"), horizontal=True)

        st.divider()
        st.subheader('Сведения о семье:')
        MARITAL_STATUS = st.radio("Семейное положение:", tuple(df.MARITAL_STATUS.unique().tolist()), horizontal=True)
        CHILD_TOTAL = st.slider("Количество детей", min_value=0, max_value=10, value=2, step=1)
        DEPENDANTS = st.slider("Количество иждивенцев", min_value=0, max_value=10, value=1, step=1)

        st.divider()
        st.subheader('Сведения о работе:')
        GEN_INDUSTRY = st.selectbox("Отрасль", tuple(df.GEN_INDUSTRY.unique().tolist()))
        JOB_DIR = st.selectbox("Направление работы", tuple(df.JOB_DIR.unique().tolist()))
        GEN_TITLE = st.selectbox("Должность", tuple(df.GEN_TITLE.unique().tolist()))
        WORK_TIME = st.number_input("Время работы в текущей должности, мес", value=36)

        st.divider()
        st.subheader('Сведения о доходах:')
        PERSONAL_INCOME = st.slider("Личный доход", min_value=0, max_value=300_000, value=15_000, step=1)
        FAMILY_INCOME = st.radio("Доход семьи", tuple(df.FAMILY_INCOME.unique().tolist()), horizontal=True)
        FL_PRESENCE_FL = st.radio("Наличие квартиры", ("Есть", "Нет"), horizontal=True)
        OWN_AUTO = st.radio("Количество автомобилей в семье", tuple(df.OWN_AUTO.unique().tolist()), horizontal=True)

        st.divider()
        st.subheader('Сведения о кредите:')
        CREDIT = st.slider("Размер кредита", min_value=1_000, max_value=100_000, value=15_000, step=1)
        TERM = st.slider("Срок кредита, мес", min_value=1, max_value=36, value=8, step=1)
        FST_PAYMENT = st.slider("Размер первоначального взноса, в % от суммы кредита", min_value=1, max_value=100,
                                value=23, step=1) * CREDIT / 100

    translatetion = {
        "Мужской": 1,
        "Женский": 0,
        "Есть": 1,
        'Нет': 0,
    }

    data = {
        "AGE": AGE,
        "GENDER": translatetion[GENDER],
        "EDUCATION": EDUCATION,
        "MARITAL_STATUS": MARITAL_STATUS,
        "CHILD_TOTAL": CHILD_TOTAL,
        "DEPENDANTS": DEPENDANTS,
        "SOCSTATUS_WORK_FL": translatetion[SOCSTATUS_WORK_FL],
        "SOCSTATUS_PENS_FL": translatetion[SOCSTATUS_PENS_FL],
        "FACT_ADDRESS_PROVINCE": FACT_ADDRESS_PROVINCE,
        "FL_PRESENCE_FL": translatetion[FL_PRESENCE_FL],
        "OWN_AUTO": OWN_AUTO,
        "CREDIT": CREDIT,
        "TERM": TERM,
        "FST_PAYMENT": FST_PAYMENT,
        "GEN_INDUSTRY": GEN_INDUSTRY,
        "GEN_TITLE": GEN_TITLE,
        "JOB_DIR": JOB_DIR,
        "WORK_TIME": WORK_TIME,
        "FAMILY_INCOME": FAMILY_INCOME,
        "PERSONAL_INCOME": PERSONAL_INCOME,
    }

    result_df = pd.DataFrame(data, index=[0])
    return result_df


def draw_model_select() -> tuple[str, str, float]:
    """
    Responsible for displaying the predictive model selection box
    Отвечает за отображение блока выбора предсказательной модели
    """
    st.divider()
    st.info('Постановка задачи: выявить для Банка-заказчика как можно больше потенциальных клиентов его '
            'будущих продуктов. Для минимизации расходов сделать это с максимально возможной точностью.')
    st.caption('Предположим, это нужно для формирования перечня получателей рекламных звонков о новом продукте Банка')
    st.divider()
    st.subheader('🔎 Выбор модели')
    st.info('Выбери предсказательную модель (они имеют разные особенности и дают разный прогноз)')
    model_name = st.radio(
        "",
        [name.title() for name in models],
        captions=[(models[name]['name'] + ': ' + models[name]['params']) for name in models], horizontal=True)

    model_type = models[model_name.lower()]['type']
    best_thr = models[model_name.lower()]['best_thr']
    return model_name, model_type, best_thr


def draw_thrash_select(best_thr) -> float:
    """
    Responsible for displaying items by selecting a probability threshold
    Отвечает за отображение элементов по выбору вероятностного порога
    """
    st.divider()
    st.subheader('🎚️ Выбор порога')
    st.info('Продемонстрируем влияние выбранного вероятностного порога на точность определения поведения клиентов')
    st.write('')
    thr_prob = st.slider('Выбери порог (шкала приведена в проценты для удобства, правильно указывать в диапазоне '
                         'от 0 до 1):', min_value=0, max_value=100, value=50, step=1)
    thr = round((thr_prob / 100), 2)
    st.write(
        "Мы подобрали для этой модели оптимальный порог для достижения максимальной точности при соблюдении заданной "
        "полноты охвата, он составляет",
        best_thr)
    st.write("Вы выбрали порог", thr)
    return thr


def draw_metrics(best_thr, thr, model_type) -> None:
    """
    Responsible for displaying the calculated quality metrics and comparing them to a benchmark
    Отвечает за отображение посчитанных метрик качества и их сравнение с эталоном
    """
    st.divider()
    st.subheader('⛳ Влияние порога на метрики качества')
    st.info(
        f'Посмотрим как изменяется качество метрик для выбранного вами порога по сравнению с эталонным решением:')
    get_metrics(threshold=thr, best_thr=best_thr, model_type=model_type)
    st.caption(
        'Но что значат эти цифры на практике?')


def draw_confusion_matrix(threshold, best_thr, model_type) -> None:
    """
    Отвечает за построение confusion matrix на основе пользовательского и эталонного порога
    Responsible for building the confusion matrix based on user and reference thresholds
    """
    _, y_thr_tuple = get_metrics_score_thr(threshold=threshold, best_thr=best_thr, model_type=model_type)

    st.divider()
    st.subheader('🫰 Влияние порога на затраты Банка')
    st.info('Продемонстрируем влияние выбранного порога на практические аспекты реальной жизни')

    col1, col2 = st.columns(2)
    pairs = zip([0, 1], ['тобой', 'нами'], [col1, col2])

    for thr_num, me, col in pairs:
        matrix = pd.DataFrame(confusion_matrix(y_test, y_thr_tuple[thr_num]))
        st.write(f'Это результат подобранного {me} порога:')
        st.dataframe(matrix.style.background_gradient(cmap='YlGnBu', axis=None))
        st.write(f'Здесь удастся правильно дозвониться {matrix[1][1]} клиентам, но мы не сделаем необходимые '
                 f'звонки {matrix[0][1]} потенциальным клиентам')
    matrix_user = confusion_matrix(y_test, y_thr_tuple[0])[1][1] + confusion_matrix(y_test, y_thr_tuple[0])[0][1]
    matrix_tuned = confusion_matrix(y_test, y_thr_tuple[1])[1][1] + confusion_matrix(y_test, y_thr_tuple[1])[0][1]
    delta = round(matrix_tuned / matrix_user, 1)
    st.write(f'Правильная для поставленной задачи тонкая настройка порога имеет свою материальную цену: нам придётся '
             f'заставить колл-центр Банка обзванивать в {delta} раз больше клиентов. Поэтому придётся считать, '
             f'что выгоднее: упустить потенциальных клиентов и недополучить доход, или нарастить себестоимость за счет '
             f'увеличения оплаты колл-центру')


def draw_choise_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обеспечивает выбор и отображение данных случайного пользователя Банка
    Provides selection and display of data from a random user of the Bank
    """
    st.divider()
    st.subheader('1️⃣ Влияние порога на единичный прогноз')

    X = df.drop(columns=['TARGET', 'AGREEMENT_RK']).reset_index(drop=True).set_index('ID')
    st.info('Нажми кнопку и выбери случайного клиента, изучи полученные результаты')
    result = st.button('Жми')
    if result:
        ind = choice(X.index.tolist())
        single_df_regular = X.loc[ind].to_frame().T
    else:
        single_df_regular = X.iloc[0].to_frame().T
    single_df_tuned = transform_df_regular_to_tuned(single_df_regular=single_df_regular)

    st.dataframe(single_df_regular)

    return single_df_regular, single_df_tuned


def draw_single_forecast(threshold, best_thr, model_type, single_df_regular, single_df_tuned) -> None:
    """
    Calculates predictions for a randomly selected random user and displays them
    Вычисляет прогнозы для случайного выбранного случайного пользователя и отображает их
    """
    if model_type == 'regular':
        single_pred, is_recommend_thr, is_recommend_best_thr = get_single_prediction(single_df_regular, threshold,
                                                                                     best_thr, model_type)
    else:
        single_pred, is_recommend_thr, is_recommend_best_thr = get_single_prediction(single_df_tuned, threshold,
                                                                                     best_thr, model_type)

    st.write('Вероятность сотрудничества этого клиента согласно прогноза модели:', round(single_pred, 2))
    text_best_thr = 'Целесообразность отработки данного клиента согласно настроенного порога:'
    text_thr = 'Целесообразность отработки данного клиента согласно установленного тобой порога:'

    if is_recommend_thr:
        st.write(text_thr, ':green[необходимо отработать]')
    else:
        st.write(text_thr, ':red[не трать время]')

    if is_recommend_best_thr:
        st.write(text_best_thr, ':green[необходимо отработать]')
    else:
        st.write(text_best_thr, ':red[не трать время]')


def draw_custom_forecast(threshold, best_thr, model_type):
    """
    Displays the virtual user design and prediction results for the virtual user
    Отображает проектирование виртуального пользователя и результаты прогноза по нему
    """
    st.divider()
    st.header('🥅 Кастомный прогноз')
    st.info('Заполни анкету виртуального клиента и получи наш прогноз о его готовности сотрудничать с Банком.')
    result_df_regular = collect_dataframe()
    result_df_tuned = transform_df_regular_to_tuned(result_df_regular)
    st.caption('Полученный портрет клиента:')
    st.dataframe(result_df_regular)
    draw_single_forecast(threshold=threshold, best_thr=best_thr, model_type=model_type,
                         single_df_regular=result_df_regular, single_df_tuned=result_df_tuned)
