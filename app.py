import pandas as pd
import streamlit as st
import config
import torch



from transformers import pipeline
from hate_speech_model import HateSpeechClassifier

import subprocess
cmd = ['python3','-m ','pip','install','--upgrade','pip']
subprocess.run(cmd)
print('Working')


# Set page title
st.title('Twitter Hate Speech Detection')

#@st.cache
# Load classification model
with st.spinner('Loading classification model...'):
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
st.subheader('Single tweet classification')

# Get sentence input, preprocess it, and convert to flair.data.Sentence format
tw = st.text_input('Tweet:')

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
            st.write(sentence)
            st.write('hate sub-cluster: ', clus_pp , ' with a Confidence Level of ', clus_cc)

    else:
        with st.spinner('Predicting...'):
            sentence

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
st.subheader('Offline Batch tweet classification')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)

  # Initialize empty dataframe
  tweet_data = pd.DataFrame({
      'tweet': [],
      'predicted-sentiment': []
  })

  # classify tweet

  for tweet in df['tweet']:

      # Skip iteration if tweet is empty
      if tweet in ('', ' '):
          continue

      # Make predictions
      class_names = ['Hate Speech', 'Normal Speech']
      sentence = sentence_prediction(tweet, model)

      # classifier.predict(sentence)
      sentiment = sentence

      max_len = 140

      if sentiment == "Hate Speech":
          #tokenizer = AutoTokenizer.from_pretrained('typeform/mobilebert-uncased-mnli')
          zero_model = 'typeform/mobilebert-uncased-mnli'

          classifier = pipeline("zero-shot-classification", model=zero_model,tokenizer=config.TOKENIZER)

          text = tweet
          candidate_labels = ['Violent', 'Offensive', 'Profane']
          result = classifier(text, candidate_labels)

          data = pd.DataFrame({'Hate Sub-clusters': result['labels'], 'Confidence Level': result['scores']})

          clus = data[data['Confidence Level'] == data['Confidence Level'].max()]

          clus_p = clus['Hate Sub-clusters'].values
          clus_pp = clus_p[0]
          clus_c = clus['Confidence Level'].values
          clus_cc = round(clus_c[0], 2)

          tweet_data = tweet_data.append({'tweet': tweet, 'predicted-sentiment': sentiment, 'hate sub-cluster': clus_pp,
                                          'confidence level': clus_cc}, ignore_index=True)
          tweet_data = tweet_data.reindex(
              columns=['tweet', 'predicted-sentiment', 'hate sub-cluster', 'confidence level'])

      else:

          non = ''
          tweet_data = tweet_data.append(
              {'tweet': tweet, 'predicted-sentiment': sentiment, 'hate sub-cluster': non, 'confidence level': non},
              ignore_index=True)
          tweet_data = tweet_data.reindex(
              columns=['tweet', 'predicted-sentiment', 'hate sub-cluster', 'confidence level'])

# As long as the query is valid (not empty or equal to '#')...

#if query != '' and query != '#':
#    with st.spinner(f'Searching for and analyzing {query}...'):


# Show query data and sentiment if available
try:
    st.write(tweet_data)
except NameError: # if no queries have been made yet
    pass