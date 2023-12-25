"""
Entry point for streamlit application
Точка входа для streamlit-приложения
"""

import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Banks Clients EDA and prediction",
    page_icon='💰',
)

# set_page_config идёт до импорта из модуля eda ввиду требований библиотеки streamlit
from eda import draw_barchart, draw_pie, draw_num_distribution, draw_categorical_distribution, draw_nun_distribution, \
    draw_pearson_correlation, draw_target_correlation
from model import draw_model_select, draw_thrash_select, draw_metrics, draw_confusion_matrix, draw_single_forecast, \
    draw_choise_df, draw_custom_forecast


def process_main_page() -> None:
    """
    Sets the required structure of elements arrangement on the main page of the application
    Задаёт необходимую структуру расположение элементов на главной странице приложения
    """
    st.title('Изучение клиентов банка и предсказание их поведения с помощью ML')
    st.image('img/bgr.png', use_column_width='auto',
             caption='Исследуем данные, предсказываем готовность клиентов к сотрудничеству: enjoy-ds@pm.me')

    tab1, tab2 = st.tabs(["Анализ", "Прогнозы"])

    with tab1:
        st.header("Exploratory Data Analysis")
        draw_barchart()
        draw_pie()
        draw_num_distribution()
        draw_categorical_distribution()
        draw_nun_distribution()
        draw_pearson_correlation()
        draw_target_correlation()

    with tab2:
        st.header('Предсказываем поведение клиентов')
        model_name, model_type, best_thr = draw_model_select()
        thr = draw_thrash_select(best_thr)
        draw_metrics(best_thr, thr, model_type)
        draw_confusion_matrix(threshold=thr, best_thr=best_thr, model_type=model_type)
        single_df_regular, single_df_tuned = draw_choise_df()
        draw_single_forecast(threshold=thr, best_thr=best_thr, model_type=model_type,
                             single_df_regular=single_df_regular,
                             single_df_tuned=single_df_tuned)
        draw_custom_forecast(threshold=thr, best_thr=best_thr, model_type=model_type)


if __name__ == '__main__':
    process_main_page()
