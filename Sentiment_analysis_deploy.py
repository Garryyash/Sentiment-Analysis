# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:54:03 2022

@author: caron
"""

#Sentiment_analysis_deploy

#%% Deployment unussually done on another pc/mobile phone

from tensorflow.keras.models import load_model
import os 
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re 


TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
CSV_URL ='https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

df=pd.read_csv(CSV_URL)


review = df['review'].values



for index, rev in enumerate(review):
    review[index] = re.sub('<.*?>',' ',rev)
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()







#%%
# to load model trained model
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))
loaded_model.summary()

#to load tokenizer
with open (TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%%
input_review = 'The movie so good, the trailer intrigues me to watch. The movie is funny. I love it'
#input_review = input('type your review here')
#preprocessing

review[index] = re.sub('<.*?>',' ',input_review)
review[index] = re.sub('[^a-zA-Z]',' ',input_review).lower().split()




tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = padded_review =pad_sequences(np.array(input_review_encoded).T,maxlen=180,
                                                             padding='post',
                                                             truncating='post')


outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)
    
print(loaded_ohe.inverse_transform(outcome))
