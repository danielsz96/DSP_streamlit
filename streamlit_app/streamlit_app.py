import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import time


@st.cache_data
def load_data():
    data = pd.read_json('/mount/src/dsp_streamlit/streamlit_app/data_senti.json')
    # data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime objects
    return data

df = load_data()


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def set_clicked():
    st.session_state.clicked = True

def run_bar():
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.5)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


with st.sidebar:
    menu = option_menu("Main Menu", ['Drug Encyclopedia', 'Sentiment Analysis', 'Map', 'Popular Drugs Ranking', 'Train model', 'Update drug dictionary', 'Inspect Data'], 
    icons=['book', 'emoji-smile-upside-down', 'map-fill', 'info', 'info', 'info',  'clipboard2-data-fill'], menu_icon="cast", default_index=1)




if menu == 'Drug Encyclopedia':
    st.title('Drug Encyclopedia')
    st.write('Choose a drug from the dropdown menu to see its description')
    selection = st.selectbox('Choose a drug', options=df['Identified Drug Type'].unique())
    # Placeholder for drug information
    if selection == 'Cocaine':
        st.write("Cocaine is a tropane alkaloid that acts as a central nervous system stimulant. As an extract, it is mainly used recreationally, and often illegally for its euphoric and rewarding effects. Wikipedia")

elif menu == 'Sentiment Analysis':
    st.title('Sentiment Analysis per Drug')
    drug_choice = st.selectbox('Choose a drug', options=df['Identified Drug Type'].unique())
    filtered_data = df[df['Identified Drug Type'] == drug_choice]
    fig = px.bar(filtered_data, x='Sentiment', y='Upvotes', color='Sentiment')
    st.plotly_chart(fig)


elif menu == 'Map':
    st.title('Map')
    # date_filter = st.slider('Select a date', value=df['Date'].min(), min_value=df['Date'].min(), max_value=df['Date'].max())
    # filtered_data = df[df['Date'] == date_filter]
    df = df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map()
    

elif menu == 'Popular Drugs Ranking':
    st.title('Popular Drugs Ranking')
    drug_count = df['Identified Drug Type'].value_counts().head(10)  # Show top 10
    st.bar_chart(drug_count)
    # st.dataframe(filter_dataframe(df))


elif menu == 'Train model':
    st.title('Train model')
    st.button('Upload File', on_click=set_clicked)
    if st.session_state.clicked:
        uploaded_file = st.file_uploader("Choose a file")
        print(uploaded_file)
        if uploaded_file is not None:
            # print(uploaded_file)
            st.write("You selected the file:", uploaded_file.name)
            st.dataframe(filter_dataframe(pd.read_csv(uploaded_file)))


    st.button('Train',type='primary', on_click=run_bar)


elif menu == 'Update drug dictionary':
    st.title('Update drug dictionary')
    st.button('Upload File', on_click=set_clicked)
    if st.session_state.clicked:
        uploaded_file = st.file_uploader("Choose a file")
        print(uploaded_file)
        if uploaded_file is not None:
            # print(uploaded_file)
            st.write("You selected the file:", uploaded_file.name)
            st.dataframe(filter_dataframe(pd.read_csv(uploaded_file)))


    st.button('Update',type='primary', on_click=run_bar)

    # progress_text = "Operation in progress. Please wait."
    # my_bar = st.progress(0, text=progress_text)

    # for percent_complete in range(100):
    #     time.sleep(0.5)
    #     my_bar.progress(percent_complete + 1, text=progress_text)
    # time.sleep(1)



elif menu == 'Inspect Data':
    st.title('Inspect Data')
    st.dataframe(filter_dataframe(df))


# Additional functionalities as per your dataset structure and requirements
