import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import time
import json
from pymongo.mongo_client import MongoClient
from ipwhois import IPWhois
from requests import get
import sys

sys.wait(300)


ip = get('https://api.ipify.org').text
whois = IPWhois(ip).lookup_rdap(depth=1)
cidr = whois['network']['cidr']
name = whois['network']['name']

print('\n')
print('Provider:  ', name)
print('Public IP: ', ip)
print('CIDRs:     ', cidr)


uri = "mongodb://danielsz:ysDC3xbgKOj863d7@ac-noqw4xe-shard-00-00.qqrkswo.mongodb.net:27017,ac-noqw4xe-shard-00-01.qqrkswo.mongodb.net:27017,ac-noqw4xe-shard-00-02.qqrkswo.mongodb.net:27017/?ssl=true&replicaSet=atlas-3kz2n9-shard-0&authSource=admin&retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)
db = client['dsp']
processed_collection = db['reddit']


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)









@st.cache_data
def load_data():
    data = pd.DataFrame(list(processed_collection.find()))
    # data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime objects
    return data


@st.cache_data
def load_drug_descriptions():
    with open('/mount/src/dsp_streamlit/streamlit_app/drug_descriptions.json', 'r') as file:
        drug_descriptions = json.load(file)
    return drug_descriptions

@st.cache_data
def load_drug_dictionary():
    with open('/mount/src/dsp_streamlit/streamlit_app/drug_dictionary.json', 'r') as file:
        drug_dictionary = json.load(file)
    return drug_dictionary


df = load_data()
drug_descriptions = load_drug_descriptions()
drug_dictionary = load_drug_dictionary()

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
    menu = option_menu("Main Menu", ['Drug Encyclopedia', 'Sentiment Analysis', 'Map', 'Popular Drugs Ranking', 'Train model', 'Update drug dictionary', 'Inspect Data', 'Info'], 
    icons=['book', 'emoji-smile-upside-down', 'map-fill', 'sort-down', 'cpu', 'book',  'clipboard2-data-fill', 'info'], menu_icon="cast", default_index=1)




if menu == 'Drug Encyclopedia':

    st.title('Drug Encyclopedia')
    selection = st.selectbox('Choose a drug from the dropdown menu to see its description', options=df['Identified Drug Type'].unique())
    # Placeholder for drug information
    st.markdown(f'## {selection}')
    st.markdown('---')
    st.write(drug_descriptions[selection])
    st.markdown('---')
    st.write('## Street names')
    keys_with_value = [key for key, value in drug_dictionary.items() if value == selection]
    st.write(str(keys_with_value))

elif menu == 'Sentiment Analysis':
    st.title('Sentiment Analysis per Drug')
    drug_choice = st.selectbox('Choose a drug', options=df['Identified Drug Type'].unique())
    filtered_data = df[df['Identified Drug Type'] == drug_choice]
    filtered_data = filtered_data.sort_values('Sentiment')
    fig = px.bar(filtered_data, x='Sentiment', y='Upvotes', color='Sentiment')
    fig.update_layout(yaxis_title='Upvotes x Sentiment', xaxis_title='Sentiment')
    st.plotly_chart(fig)

    sentiment_data = filtered_data.groupby('Date')['Sentiment_class'].mean().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sentiment_data['Date'], y=sentiment_data['Sentiment_class'], name='Sentiment'))
    fig2.update_layout(title='Mean sentiment over time', xaxis_title='Date', yaxis_title='Sentiment')
    st.plotly_chart(fig2)


elif menu == 'Map':
    st.title('Map')
    # date_filter = st.slider('Select a date', value=df['Date'].min(), min_value=df['Date'].min(), max_value=df['Date'].max())
    # filtered_data = df[df['Date'] == date_filter]
    df = df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map()
    

elif menu == 'Popular Drugs Ranking':
    st.title('Popular Drugs Ranking')
    drug_count = df['Identified Drug Type'].value_counts().head(10)  # Show top 10
    fig = go.Figure(data=[go.Bar(x=drug_count.index, y=drug_count.values)])
    fig.update_layout(title='Ranking by mentions', xaxis_title='Drug Type', yaxis_title='Mention count')
    st.plotly_chart(fig)
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


elif menu == 'Info':
    st.title('Info')
    st.markdown('---')
    st.write('## Dataflow')
    st.image('/mount/src/dsp_streamlit/streamlit_app/flow.png')
    st.markdown('---')
    st.write('## Model info')
    st.write('''
            ### Named entity recognition: 
             [bert-base-uncased] finetuned on Reddit posts - annotated with RegEx


            ### Sentiment analysis:
             [cardiffnlp/twitter-roberta-base-sentiment]
             ''')


# Additional functionalities as per your dataset structure and requirements
