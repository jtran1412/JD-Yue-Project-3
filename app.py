import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.title("Yelp Review Classifier")

yelp_df_1_5 = st.cache(pd.read_csv)("yelp_df_1_5.csv")

punct = string.punctuation
stopwords = stopwords.words('english')

def message_cleaning(message):
    punct_removed = [char for char in message if char not in punct]
    punct_removed_join = ''.join(punct_removed)
    punct_removed_join_split = punct_removed_join.split()
    punct_stopwords_removed = [word for word in punct_removed_join_split if word.lower() not in stopwords]
    return punct_stopwords_removed

vectorizer = CountVectorizer(analyzer = message_cleaning)

yelp_vectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])

X = yelp_vectorizer
y = yelp_df_1_5['stars'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

NB_classifier = MultinomialNB()

NB_classifier.fit(X_train, y_train)

y_predict_train = NB_classifier.predict(X_train)

testing_entry = st.text_input("Enter your review:")

testing = [testing_entry]

testing_vectorizer = vectorizer.transform(testing)

test_predict = NB_classifier.predict(testing_vectorizer)

st.write("Predicted rating: ", test_predict[0])