"""
Module responsible for content and display of the EDA tab in streamlit application
Модуль, отвечающий за содержание и отображение вкладки "Анализ" в streamlit-приложении
"""

from imports import (st, pd, px, df, charts_dict, charts_texts, pie_dict, pie_texts, numeric_columns, target_distr_dict,
                     target_distr_texts)


def draw_barchart():
    """
    Controls the block that draws bar charts for individual data categories
    Управляет отображением блока, рисующего бар-чарты для отдельных категорий данных
    """
    st.divider()

    columns = st.radio(
        "Выбери категорию для построения обычной диаграммы:",
        [rus_names for rus_names in charts_dict.keys()]
    )

    st.subheader(f'🎢 Распределение заёмщиков в разрезе категории "{columns}"')
    st.bar_chart(df, x=charts_dict[columns], y='ID')
    st.caption(charts_texts[columns])
    # with st.expander('Прочесть комментарий'):
    #     st.write(expanders[columns])


def draw_pie():
    """
    Controls the block that draws a pie chart for individual data categories
    Управляет отображением блока, рисующего круговые диаграммы для отдельных категорий данных
    """
    st.divider()

    columns = st.radio(
        "Выбери категорию для построения круговой диаграммы:",
        [rus_names for rus_names in pie_dict.keys()]
    )
    st.subheader(f'🍰 Распределение заёмщиков в разрезе категории "{columns}"')

    category = pie_dict[columns]
    data_frame = pd.DataFrame(df[category].value_counts(), columns=['count'])
    fig = px.pie(data_frame=data_frame, values='count', names=data_frame.index)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.caption(pie_texts[columns])


def draw_num_distribution():
    """
    Controls the display of a block drawing the distribution of numeric columns
    Управляет отображением блока, рисующего распределение числовых столбцов
    """
    st.divider()
    st.subheader('🔢 Характеристики распределения числовых столбцов')
    st.dataframe(
        df.drop(
            columns=numeric_columns
        )
        .describe()
    )


def draw_categorical_distribution():
    """
    Controls the display of a block drawing the distribution of categorical columns
    Управляет отображением блока, рисующего распределение категориальных столбцов
    """
    st.subheader('📌 Характеристики распределения категориальных столбцов')
    st.dataframe(df.describe(include='object'))
    st.caption(
        'Любопытно было взглянуть на портрет среднего заёмщика: 36 летний продавец из Кемеровской области с '
        'семейным доходом до 20 тысяч рублей, имеющий одного ребенка и берущий микрозайм в 15 тысяч рублей на '
        '8 месяцев.'
    )


def draw_nun_distribution():
    """
    Controls the display of the block showing missing columns
    Управляет отображением блока, показывающим отсутствующие столбцы
    """
    st.divider()
    st.subheader('🫗 Пропуски в данных')
    st.dataframe(df.isna().sum())
    st.caption(
        'В данных нет пропусков, это обусловлено самой сборкой датасета, что описано в ноутбуке. Ну вот, '
        'зачем-то ещё раз убедились в этом.'
    )


def draw_pearson_correlation():
    """
    Controls the display of the block that builds the Pearson correlation
    Управляет отображением блока, который строит корреляцию Пирсона
    """

    st.divider()
    st.subheader('🔗 Корреляция Пирсона')
    st.dataframe(df.select_dtypes('float64').corr().style.background_gradient(cmap='coolwarm'))
    st.caption(
        'Есть сильная корреляция между доходом заёмщика, его первым платежом и размером кредита, '
        'и это совершенно понятно. Но вот корреляция между числовыми признаками и целевой переменной отсутствует '
        'практически полностью. Мне кажется, ключевыми "рабочими" фичами при построении предсказательных моделей станут'
        ' категориальные признаки'
    )


def draw_target_correlation():
    """
    Controls the display of the block that plots the scatter plots of the target variable
    Управляет отображением блока, который строит диаграммы рассеивания целевой переменной
    """
    st.divider()

    columns = st.radio(
        "Выбери категории для построения диаграммы рассеивания:",
        [rus_names for rus_names in target_distr_dict.keys()]
    )
    st.subheader(
        f'🎯 Распределение целевой переменной в зависимости от возраста заёмщика и значений категории "{columns}"')

    category = target_distr_dict[columns]
    st.scatter_chart(df[[category, 'AGE', "TARGET"]], x=category, y='AGE', color='TARGET', use_container_width=True)
    st.caption(target_distr_texts[columns])
