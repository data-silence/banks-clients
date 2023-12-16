from imports import st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Banks Clients EDA and prediction",
    page_icon='💰',

)

from eda import draw_barchart, draw_pie, num_distribution, categorical_distribution, nun_distribution, \
    pirson_correlation


def process_main_page() -> None:
    """
    Sets the required structure of elements arrangement on the main page of the application
    Задаёт необходимую структуру расположение элементов на главной странице приложения
    """
    st.title('Изучение клиентов банка и предсказание их поведения с помощью ML')
    st.image('img\\bgr.png', use_column_width='auto',
             caption='Исследуем данные, предсказываем готовность клиентов к сотрудничеству: enjoy-ds@pm.me')

    tab1, tab2, tab3 = st.tabs(["Анализ", "Прогнозы", "Лучшая модель"])

    with tab1:
        st.header("Exploratory Data Analysis")
        draw_barchart()
        draw_pie()
        num_distribution()
        categorical_distribution()
        nun_distribution()
        pirson_correlation()

    with tab2:
        st.title('⛏️ Идёт разработка...')

    with tab3:
        st.title('🔨 Идёт разработка...')


if __name__ == '__main__':
    process_main_page()
