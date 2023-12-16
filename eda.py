from imports import st, pd, px, df, charts_dict, charts_texts, pie_dict, pie_texts, numeric_columns


def draw_barchart():
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


def num_distribution():
    st.divider()
    st.subheader('🔢 Характеристики распределения числовых столбцов')
    st.dataframe(
        df.drop(
            columns=numeric_columns
        )
        .describe()
    )


def categorical_distribution():
    st.divider()
    st.subheader('📌 Характеристики распределения категориальных столбцов')
    st.dataframe(df.describe(include='object'))


def nun_distribution():
    st.divider()
    st.subheader('🫗 Пропуски в данных')
    st.dataframe(df.isna().sum())


def pirson_correlation():
    st.divider()
    st.subheader('🔗 Корреляция Пирсона')
    st.dataframe(df.select_dtypes('float64').corr().style.background_gradient(cmap='coolwarm'))
