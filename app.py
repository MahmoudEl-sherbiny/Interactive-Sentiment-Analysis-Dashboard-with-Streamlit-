import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# File path
DATA_URL = "./Tweets.csv"

# Title and descriptions
st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used to analyze sentiments of tweets 🐦")
st.sidebar.markdown("This application is a Streamlit dashboard used to analyze sentiments of tweets 🐦")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    return df

data = load_data()

# Show random tweet
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))

# Cache the random tweet so it doesn't change unless the sentiment changes
if 'last_sentiment' not in st.session_state or st.session_state.last_sentiment != random_tweet:
    st.session_state.last_sentiment = random_tweet
    st.session_state.random_tweet_text = data.query("airline_sentiment == @random_tweet")["text"].sample(1).iat[0]

st.sidebar.markdown(st.session_state.random_tweet_text)

# Sentiment count plot
st.sidebar.subheader("Number of tweets by sentiment")
sentiment_plot_type = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts().reset_index()
sentiment_count.columns = ['Sentiment', 'Tweets']

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if sentiment_plot_type == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
    st.plotly_chart(fig)

# Tweet locations by time
st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
filtered_data = data[data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Hide", True, key='2'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown(f"{len(filtered_data)} tweets between {hour}:00 and {(hour + 1)%24}:00")
    st.map(filtered_data)

    if st.sidebar.checkbox("Show raw data", False):
        st.write(filtered_data)

# Tweet count per airline
st.sidebar.subheader("Total number of tweets for each airline")
airline_plot_type = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='3')
airline_count = data['airline'].value_counts().reset_index()
airline_count.columns = ['Airline', 'Tweets']

if not st.sidebar.checkbox("Hide", True, key='4'):
    st.subheader("Total number of tweets for each airline")
    if airline_plot_type == 'Bar plot':
        fig_airline = px.bar(airline_count, x='Airline', y='Tweets', color='Tweets', height=500)
    else:
        fig_airline = px.pie(airline_count, values='Tweets', names='Airline')
    st.plotly_chart(fig_airline)

# Function to get sentiment count per airline
@st.cache_data
def get_airline_sentiment_distribution(airline):
    df = data[data['airline'] == airline]
    counts = df['airline_sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Tweets']
    return counts

# Sentiment breakdown per airline
st.sidebar.subheader("Breakdown airline by sentiment")
airline_choices = st.sidebar.multiselect('Pick airlines', data['airline'].unique(), key='5')

if airline_choices:
    st.subheader("Breakdown airline by sentiment")
    viz_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot'], key='6')
    fig = make_subplots(
        rows=1,
        cols=len(airline_choices),
        subplot_titles=airline_choices,
        specs=[[{'type': 'domain'}] * len(airline_choices)] if viz_type == 'Pie chart' else None
    )

    for idx, airline in enumerate(airline_choices):
        sentiment_data = get_airline_sentiment_distribution(airline)
        if viz_type == 'Bar plot':
            fig.add_trace(
                go.Bar(x=sentiment_data['Sentiment'], y=sentiment_data['Tweets'], showlegend=False),
                row=1, col=idx + 1
            )
        else:
            fig.add_trace(
                go.Pie(labels=sentiment_data['Sentiment'], values=sentiment_data['Tweets'], showlegend=True),
                row=1, col=idx + 1
            )

    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)

# Histogram across airlines by sentiment
if airline_choices:
    choice_data = data[data['airline'].isin(airline_choices)]
    fig_hist = px.histogram(
        choice_data,
        x='airline',
        y='airline_sentiment',
        histfunc='count',
        color='airline_sentiment',
        facet_col='airline_sentiment',
        height=600, width=800
    )
    st.plotly_chart(fig_hist)

# Word cloud
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Hide", True, key='7'):
    st.subheader(f'Word cloud for {word_sentiment} sentiment')
    df_wc = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df_wc['text'])
    cleaned_words = ' '.join([
        word for word in words.split()
        if 'http' not in word and not word.startswith('@') and word != 'RT'
    ])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(cleaned_words)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
