import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from datasets import load_dataset

nltk.download('stopwords')
nltk.download('punkt')

data = load_dataset("MartinThoma/wili_2018")
df = pd.DataFrame(data['train'])

print(df.head())

stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

df['cleaned_text'] = df['sentence'].apply(preprocess_text)
print(df[['sentence', 'cleaned_text', 'label']].head())

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"Training data shape: {X_train_vectorized.shape}")
print(f"Testing data shape: {X_test_vectorized.shape}")

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

def predict_language(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

new_texts = [
    "Hello, how are you?",
    "Bonjour, comment ça va?",
    "Hola, ¿cómo estás?",
    "ब्रेकिंग न्यूज़, वीडियो, ऑडियो और फ़ीचर."
]

for text in new_texts:
    lang_prediction = predict_language(text)
    print(f'Text: "{text}" is predicted to be in language: {lang_prediction}')
