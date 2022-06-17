# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:39:22 2022

@author: caron
"""

#%% Imports

import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


#%% Constant
CSV_URL ='https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'


#%%
# EDA
# Step 1) Data Loading

df=pd.read_csv(CSV_URL)
df_copy = df.copy() #back-up


# Step2) Data Inspection
df.head(10)
df.tail(10)
df.info
df.describe()


df['sentiment'].unique() #to get the unique target
df['review'][5]
df['sentiment'][5]
df.duplicated().sum() # number of duplicated data
df[df.duplicated()]


# Step 3) Data Cleaning

df = df.drop_duplicates() #remove duplicated data

# remove html tags
'<br /> djwiajdwdiwjwdwijdai <br />'.replace('<br />', ' ')

review = df['review'].values
sentiment = df['sentiment'].values


for index, rev in enumerate(review):
    review[index] = re.sub('<.*?>',' ',rev)
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()


# Step 4) Features selection
# nothing to select



# Step 5) Data Preprocessing

#   1 Convert into lower case
            #Done
#   2 Tokenization

vocab_size = 10000
oov_token = 'OOV'


tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review)
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(review) # to convert into numbers

#   3) Padding & truncating

length_of_review = [len(i) for i in train_sequences]
np.median(length_of_review)

max_len = 180

padded_review =pad_sequences(train_sequences,maxlen=max_len,padding='post',truncating='post')
 
#   4) One Hot Encoding for the target

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

 #   5) Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_review,sentiment,test_size=0.3,random_state=123)
 

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
#%% Model Development
# use LSTM layers,dropout, dense ,

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , Dropout
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional,Embedding

embedding_dim = 64

model=Sequential()
model.add(Input(shape=(180))) #input
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
#model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,'softmax'))
model.summary()

plot_model(model)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=128)

import matplotlib.pyplot as plt

hist.history.keys()
plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()


#%% Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

y_true = y_test

y_pred = model.predict(X_test)

y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1) 

print(classification_report(y_true,y_pred))
# model score
print(accuracy_score(y_true,y_pred))

print(confusion_matrix(y_true,y_pred))


#%% Model saving
import os
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open (TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

import pickle
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open (OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#%% Discussion
# Discuss your results
# Model achieved around 84% accuracy during training
# Recall and fi-score reports 87% and 84% respectively
# However the model starts to overfit after 2nd epoch
# EarlyStopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting
# Trying with different Model architecture for example BERT model, Transformer   
   
# 1) resuts ---> Discuss your results
# 2) give suggestion on how to improve the model
# 3) Gather evidence on what went wrong during training/model development


#%% Deployment unussually done on another pc/mobile phone
from tensorflow.keras.models import load_model
import os

#load model
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))
loaded_model.summary()

#load tokenizer
with open (TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%%

input_review = 'The movie so good, the trailer intrigues me to watch. The movie is funny. I love it'
input_review = input('type your review here')
#preprocessing

review[index] = re.sub('<.*?>',' ',input_review)
review[index] = re.sub('[^a-zA-Z]',' ',input_review).lower().split()

from tensorflow.keras.preprocessing.text import tokenizer_from_json



tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = padded_review =pad_sequences(np.array(input_review_encoded).T,maxlen=180,
                                                             padding='post',
                                                             truncating='post')


outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)
    
print(loaded_ohe.inverse_transform(outcome))




