import pandas as pd

from imports import st, prediction_tuned, prediction_regular, y_test, models, df, choice, ohe_enc, scaler, \
    model_regular, model_tuned

# from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, \
    confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer


# from mlxtend.plotting import plot_confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sn
# import altair as alt
# from sklearn.metrics import confusion_matrix

def get_metrics_score(y_test, y_pred) -> dict:
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def get_metrics_score_thr(threshold: float, best_thr: float, model_type: str = 'regular') -> tuple[dict, tuple]:
    pred_probability = prediction_regular if model_type == 'regular' else prediction_tuned
    y_thr = pred_probability > threshold
    y_best_thr = pred_probability > best_thr
    metrics = {'user': get_metrics_score(y_test, y_thr), 'best': get_metrics_score(y_test, y_best_thr)}
    y_thr_tuple = (y_thr, y_best_thr)
    return metrics, y_thr_tuple


def get_metrics(threshold: float, best_thr: float, model_type: str = 'regular') -> None:
    metrics, _ = get_metrics_score_thr(threshold=threshold, best_thr=best_thr, model_type=model_type)
    col1, col2, col3, col4 = st.columns(4)
    mets_name = list(metrics['user'].keys())
    pairs = zip(mets_name, [col1, col2, col3, col4])

    for mets, col in pairs:
        col.metric(mets, metrics['user'][mets], round((metrics['user'][mets] - metrics['best'][mets]), 4))


def draw_model_select() -> tuple[str, str, float]:
    st.write('Постановка задачи: определить для нашего заказчика-Банка как можно больше потенциальных клиентов его '
             'будущих продуктов, и с максимально возможной точностью.')
    st.caption('Предположим, это делается для совершения рекламных звонков о новом продукте')
    model_name = st.radio(
        "Выбери модель",
        [name.title() for name in models],
        captions=[(models[name]['name'] + ': ' + models[name]['params']) for name in models], horizontal=True)

    model_type = models[model_name.lower()]['type']
    best_thr = models[model_name.lower()]['best_thr']
    return model_name, model_type, best_thr


def draw_thrash_select(best_thr) -> float:
    st.divider()
    st.subheader('🎚️ Выбор порога')
    st.caption('Демонстрация влияния уровня вероятностного порога на правильность определения поведения клиентов')
    st.write('')
    thr_prob = st.slider('Выбери порог (шкала приведена в проценты для удобства, правильно указывать в диапазоне '
                         'от 0 до 1):', min_value=0, max_value=100, value=int(best_thr * 100), step=1)
    thr = round((thr_prob / 100), 2)
    st.write(
        "Мы подобрали для этой модели оптимальный порог для достижения максимальной точности при соблюдении заданной полноты охвата, он составляет",
        best_thr)
    st.write("Вы выбрали порог", thr)
    return thr


def draw_metrics(best_thr, thr, model_type) -> None:
    st.write('')
    st.write(f'Посмотрим как изменяется качество метрик для выбранного вами порога по сравнению с эталонным решением:')
    get_metrics(threshold=thr, best_thr=best_thr, model_type=model_type)
    st.caption(
        'Но что значат эти цифры на практике? Покажем влияние порога на фактический результат решения нашей задачи')


def draw_confusion_matrix(threshold, best_thr, model_type) -> None:
    _, y_thr_tuple = get_metrics_score_thr(threshold=threshold, best_thr=best_thr, model_type=model_type)

    st.divider()
    st.subheader('🫰 Влияние порога на затраты Банка')

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
             f'что выгодее: упустить потенциальных клиентов и недополучить доход, или нарастить себестоимость за счет '
             f'увеличения оплаты колл-центру')


def draw_single_forecast(threshold, best_thr, model_type) -> None:
    st.divider()
    st.subheader('1️⃣ Влияние порога на единичный прогноз')

    X = df.drop(columns=['TARGET', 'AGREEMENT_RK']).reset_index(drop=True).set_index('ID')
    st.write('')
    result = st.button('Нажми, чтобы выбрать случайного клиента и получить прогноз по нему')
    if result:
        ind = choice(X.index.tolist())
        single_df = X.loc[ind].to_frame().T
    else:
        single_df = X.iloc[0].to_frame().T

    st.dataframe(single_df)

    df_new = pd.DataFrame(ohe_enc.transform(single_df))
    df_new = pd.DataFrame(scaler.transform(df_new), columns=df_new.columns, index=df_new.index)

    # st.dataframe(df_new)

    if model_type == 'regular':
        predict_model = model_regular
        single_pred_positive = predict_model.predict_proba(single_df)[:, 1][0]
    else:
        predict_model = model_tuned
        single_pred_positive = predict_model.predict_proba(df_new)[:, 1][0]

    st.write('Вероятность сотрудничества этого клиента согласно прогноза модели:', round(single_pred_positive, 2))
    text_thr = 'Целесообразность отработки данного клиента согласно установленного тобой порога:'
    if single_pred_positive > threshold:
        st.write(text_thr, ':green[необходимо отработать]')
    else:
        st.write(text_thr, ':red[не трать время]')
    text_best_thr = 'Целесообразность отработки данного клиента согласно настроенного порога:'
    if single_pred_positive > best_thr:
        st.write(text_best_thr, ':green[необходимо отработать]')
    else:
        st.write(text_best_thr, ':red[не трать время]')
