import streamlit as st
import spacy
from spacy import displacy
from spacy_streamlit import visualize_spans

import pickle
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("french")


#Initialize a TfidfVectorizer
f = open('./Classification_Models/ML_Model/tfidf_vectorizer.pkl', 'rb')
tfidf_vectorizer = pickle.load(f)
f.close()

f = open('./Classification_Models/ML_Model/model_95.81.pkl', 'rb')
classifier_model = pickle.load(f)
f.close()

# Load Spacy Model
spacy_model = spacy.load("./spacy-model/model-best")

def preprocess_text(text):
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """
    
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("french")]
    text = " ".join(tokens)
    text = text.lower().strip()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def Predict(model, article):
    article_preprocessed = preprocess_text(article)
    article_preprocessed = ' '.join([stemmer.stem(word) for word in article_preprocessed.split()])
    tfidf_article = tfidf_vectorizer.transform([article_preprocessed])
    prediction = model.predict(tfidf_article)
    if prediction[0] == 1:
        return 'Zone'
    return 'Not Zone'


st.title("Zone text classification Web Application")
article = st.text_input("Entre an article", "")

if article != '':
    if st.button('Predict'):
        prediction = Predict(classifier_model, article)
        st.write(prediction)
        if prediction == 'Zone':
            doc = spacy_model(article)
            ent_html = displacy.render(doc, style="ent", jupyter=False)
            st.markdown(ent_html, unsafe_allow_html=True)