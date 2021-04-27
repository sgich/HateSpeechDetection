# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
import time
import tweepy
import datetime
from datetime import datetime, date, time

import plotly.express as px
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import config
import torch
from transformers import pipeline
from hate_speech_model import HateSpeechClassifier

#---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

image = Image.open('4.PNG')

st.image(image, width = None)

df = pd.read_csv('data/merged_hatespeech_dataset - merged_hatespeech_dataset.csv')

df['hate_speech(1=hspeech, 0=nohspeech)'] = np.where(df['hate_speech(1=hspeech, 0=nohspeech)']==1,'Hate speech','Normal speech')

# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.beta_columns((2,1))

#---------------------------------#
# Sidebar - Main panel
col1.header('Select options')

## Sidebar - Currency price unit
region = df['location'].unique()
selected_region = col1.selectbox('Select region', region)

## Sidebar - Start and End date
start_date = col1.date_input('Start date')
start_date = pd.to_datetime(start_date)
end_date = col1.date_input('End date')
end_date = pd.to_datetime(end_date)


# date_range = col1.date_input('Date Range',value=(datetime(2020, 1, 1), datetime(2030, 1, 1)), help="choose a range or click same day twice") #,datetime(2021, 4, 17),datetime(2021, 4, 27))

#end_start = col1.date_input('End Date',datetime(2021, 4, 17),datetime(2021, 4, 27))
#d5 = col1.date_input("date range without default", [datetime(2019, 7, 6), datetime(2019, 7, 8)])
#col1.write(d5)

#st.title('Twitter hatespeech detection tool')
st.markdown("""
This tool extracts tweets from twitter and classifies them as **hate speech or non-hatespeech**!
""")
#---------------------------------#
# About
expander_bar_1 = st.beta_expander("About this tool")
expander_bar_1.markdown("""
In an increasingly digital era where online social interactions are considered part of the social context, it is proving inevitable that machine learning should be used to protect people from harmful content. This has been evidenced by the multitude of instances where hate speech propagated online has led to physical injury and loss of lives across the world. Government institutions should now consider online interactions as spaces where potential crimes may occur just like in the physical world.

This tool identifies hatespeech as tweets that can be in following three formal classes:
* **HATE:** : This class contains tweets which highlight negative attributes or deficiencies of certain groups of individuals. This class includes hateful comments towards individuals based on race, political opinion, sexual orientation, gender, social status, health condition, etc.
* **OFFN:** : This class contains tweets which are degrading, dehumanizing or insulting towards an individual. It encompasses cases of threatening with violent acts.
* **PRFN:** : This class contains tweets with explicit content, profane words or unacceptable language in the absence of insults and abuse. This typically concerns the usage of swearwords and cursing.

Political hate speech is the greatest area of concern in regards to Kenya and thus this will be the area of focus for this tool.
""")
#---------------------------------#

# Scraping of tweets
expander_bar_2 = st.beta_expander("Search/load tweets")

#@st.cache
# Load classification model
with st.spinner('Loading...'):
    model = HateSpeechClassifier()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


@st.cache(allow_output_mutation=True)
def sentence_prediction(tw, model):
    tokenizer = config.TOKENIZER
    max_len = 140
    review = str(tw)

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length"

    )



    class_names = ['Normal Speech','Hate Speech']

    input_ids = inputs['input_ids']
    mask = inputs['attention_mask']

    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    input_ids = input_ids.to(device, dtype=torch.long)
    attention_mask = attention_mask.to(device, dtype=torch.long)

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask
                    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    out = outputs[0][0]

    hate_prediction = float(out)

    if hate_prediction >= 0.5:
        return f"{class_names[1]}"
    else:
        return f"{class_names[0]}"



### SINGLE TWEET CLASSIFICATION ###
expander_bar_2.subheader('Single tweet classification')

# Get sentence input, preprocess it, and convert to flair.data.Sentence format
tw = expander_bar_2.text_input('Tweet:')

if tw != '':
    # Predict tweet
    sentence = sentence_prediction(tw, model)

    # Show prediction
    #with st.spinner('Predicting...'):
        #sentence

    if sentence == "Hate Speech":

        zero_model = 'typeform/mobilebert-uncased-mnli'

        classifier = pipeline("zero-shot-classification", model=zero_model,tokenizer=config.TOKENIZER)

        text = tw
        candidate_labels = ['Violent', 'Offensive', 'Profane']
        result = classifier(text, candidate_labels)

        data = pd.DataFrame({'Hate Sub-clusters': result['labels'], 'Confidence Level': result['scores']})

        clus = data[data['Confidence Level'] == data['Confidence Level'].max()]

        clus_p = clus['Hate Sub-clusters'].values
        clus_pp = clus_p[0]
        clus_c = clus['Confidence Level'].values
        clus_cc = round(clus_c[0], 2)


        #print('hate sub-cluster: ', clus_pp ,' with a Confidence Level of ', clus_cc)

            #f"{'hate sub-cluster': clus_pp,'Confidence Level': clus_cc}"

        with st.spinner('Predicting...'):

            speech = f"**{sentence}**"
            subclust = f"**Hate sub-cluster: {clus_pp}  with a Confidence Level of {clus_cc}**"
            #st.markdown(speech)
            expander_bar_2.write(speech)
            #st.markdown(subclust)
            expander_bar_2.write(subclust)



    else:
        with st.spinner('Predicting...'):
            speech = f"**{sentence}**"
            #st.markdown(speech)
            expander_bar_2.write(speech)

        #st.write(alt.Chart(data).mark_bar().encode(
           # x='Confidence Level',
          #  y=alt.X('Hate Sub-clusters', sort=None),
         #   color='Hate Sub-clusters'

        #).configure_axis(
        #    grid=False
        #).properties(
        #    width=500,
        #    height=150
        #)
       # )
       # st.write(out)


### TWEET SEARCH AND CLASSIFY ###
expander_bar_2.subheader('Offline Batch tweet classification')

# Initialize empty dataframe
tweet_data = pd.DataFrame({
    'tweet': [],
    'predicted-sentiment': [],
    'location':[],
    'tweet_date':[]
  })

uploaded_file = expander_bar_2.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  expander_bar_2.write(df)


  # classify tweet

  for index, row in df.iterrows():
#   for tweet,location,tweet_date in df[['tweet','location','tweet_date']]:

      # Skip iteration if tweet is empty
      if row['tweet'] in ('', ' '):
          continue

      # Make predictions
      class_names = ['Hate Speech', 'Normal Speech']
      sentence = sentence_prediction(row['tweet'], model)

      # classifier.predict(sentence)
      sentiment = sentence

      max_len = 140

      if sentiment == "Hate Speech":
          #tokenizer = AutoTokenizer.from_pretrained('typeform/mobilebert-uncased-mnli')
          zero_model = 'typeform/mobilebert-uncased-mnli'

          classifier = pipeline("zero-shot-classification", model=zero_model,tokenizer=config.TOKENIZER)

          text = row['tweet']
          candidate_labels = ['Violent', 'Offensive', 'Profane']
          result = classifier(text, candidate_labels)

          data = pd.DataFrame({'Hate Sub-clusters': result['labels'], 'Confidence Level': result['scores']})

          clus = data[data['Confidence Level'] == data['Confidence Level'].max()]

          clus_p = clus['Hate Sub-clusters'].values
          clus_pp = clus_p[0]
          clus_c = clus['Confidence Level'].values
          clus_cc = round(clus_c[0], 2)

          tweet_data = tweet_data.append({'tweet': row['tweet'], 'predicted-sentiment': sentiment, 'hate sub-cluster': clus_pp,
                                          'confidence level': clus_cc, 'location':row['location'],'tweet_date': row['tweet_date']}, ignore_index=True)
          tweet_data = tweet_data.reindex(
              columns=['tweet', 'predicted-sentiment', 'hate sub-cluster', 'confidence level', 'location','tweet_date'])

      else:

          non = ''
          tweet_data = tweet_data.append({'tweet': row['tweet'], 'predicted-sentiment': sentiment, 'hate sub-cluster': non, 'confidence level': non, 'location':row['location'],'tweet_date': row['tweet_date']}, ignore_index=True)
          tweet_data = tweet_data.reindex(
              columns=['tweet', 'predicted-sentiment', 'hate sub-cluster', 'confidence level', 'location','tweet_date'])

# As long as the query is valid (not empty or equal to '#')...

#if query != '' and query != '#':
#    with st.spinner(f'Searching for and analyzing {query}...'):


# Show query data and sentiment if available
try:
    tweet_data['tweet_date'] =pd.to_datetime(tweet_data['tweet_date'])
    tweet_data_filtered = tweet_data[(tweet_data['location']==selected_region) & (tweet_data['tweet_date']>=start_date) & (tweet_data['tweet_date']<=end_date)]
    expander_bar_2.write(tweet_data_filtered)
except NameError: # if no queries have been made yet
    pass
#---------------------------------#

# Overview of extracted tweets
tweet_data['tweet_date'] =pd.to_datetime(tweet_data['tweet_date'])
tweet_data_filtered = tweet_data[(tweet_data['location']==selected_region) & (tweet_data['tweet_date']>=start_date) & (tweet_data['tweet_date']<=end_date)]
expander_bar_3 = st.beta_expander("Visual overview of loaded tweets")
sentiment_count = tweet_data_filtered['predicted-sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Tweets':sentiment_count.values})

# region_count = df['location'].value_counts()
# region_count = pd.DataFrame({'Region':region_count.index,'Tweets':region_count.values})
if len(sentiment_count) == 0:
    expander_bar_3.markdown('There are no visuals at the moment...... Please load data to show some visuals')
else:
    fig_1 = px.bar(sentiment_count,x='Sentiments',y='Tweets',color='Tweets',height=500)
    expander_bar_3.plotly_chart(fig_1)
    
    fig_2 = px.pie(sentiment_count,values='Tweets',names='Sentiments')
    expander_bar_3.plotly_chart(fig_2)   
# fig_3 = px.bar(region_count,x='Region',y='Tweets',color='Tweets',height=500)
# expander_bar_3.plotly_chart(fig_3)

#expander_bar_3.table()

#---------------------------------#
# Hate speech tweets
expander_bar_3 = st.beta_expander("View hatespeech tweets")
df_hatespeech = tweet_data_filtered[tweet_data_filtered['predicted-sentiment']=='Hate Speech']
if len(df_hatespeech) == 0:
    expander_bar_3.markdown('Nothing to show here since hate speech has not been detected in the set of uploaded tweets')
else:
    expander_bar_3.dataframe(df_hatespeech[['tweet','predicted-sentiment']])
#---------------------------------#

# Non-hatespeech tweets
expander_bar_4 = st.beta_expander("View normal text tweets")
df_normalspeech = tweet_data_filtered[tweet_data_filtered['predicted-sentiment']=='Normal Speech']
if len(df_normalspeech) == 0:
    expander_bar_4.markdown('Nothing to show here since normal speech has not been detected in the set of uploaded tweets')
else:
    expander_bar_4.dataframe(df_normalspeech[['tweet','predicted-sentiment']])
#---------------------------------#

#---------------------------------#

#---------------------------------#
# Hate speech words
st.set_option('deprecation.showPyplotGlobalUse', False)
expander_bar_5 = st.beta_expander("Hate speech key words")
if len(df_hatespeech) == 0:
    expander_bar_5.markdown('Nothing to show here since hate speech has not been detected in the set of uploaded tweets')
else:
    words = " ".join(df_hatespeech["tweet"])
    processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    expander_bar_5.pyplot()
#---------------------------------#