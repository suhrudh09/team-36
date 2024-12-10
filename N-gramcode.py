import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from collections import defaultdict
from datasets import load_dataset
import string

nltk.download('stopwords')

data = load_dataset("MartinThoma/wili_2018")
df = pd.DataFrame(data['train'])

stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

df['cleaned_text'] = df['sentence'].apply(preprocess_text)

def generate_ngrams(text, n):
    tokens = text.split()
    return list(ngrams(tokens, n))

df['quadgrams'] = df['cleaned_text'].apply(lambda x: generate_ngrams(x, 4))

print(df[['cleaned_text', 'quadgrams']].head())

quadgram_freqs = defaultdict(lambda: defaultdict(int))

for index, row in df.iterrows():
    for qg in row['quadgrams']:
        quadgram_freqs[row['label']][qg] += 1

def predict_language(text):
    cleaned_text = preprocess_text(text)
    quadgrams = generate_ngrams(cleaned_text, 4)
    
    score_dict = defaultdict(int)
    
    for qg in quadgrams:
        for lang, freqs in quadgram_freqs.items():
            score_dict[lang] += freqs[qg]
    
    predicted_language = max(score_dict, key=score_dict.get, default="Unknown Language")
    
    return predicted_language

new_texts = [
    "ब्रेकिंग न्यूज़, वीडियो, ऑडियो और फ़ीचर",
]

for text in new_texts:
    lang_prediction = predict_language(text)
    print(f'Text: "{text}" is predicted to be in language: {lang_prediction}')
