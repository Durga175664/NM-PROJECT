import streamlit as st
import pickle
import string

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Simple stopwords list
stop_words = set([
    'the', 'is', 'in', 'and', 'he', 'she', 'it', 'on', 'for', 'to', 'of', 'a', 'an', 'as', 'with', 'has',
    'this', 'that', 'was', 'be', 'from', 'at', 'by', 'but', 'or', 'not', 'are'
])

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

st.title("📰 Fake News Detector")

news = st.text_area("Enter news content:")

if st.button("Predict"):
    cleaned = preprocess(news)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    st.subheader("Prediction:")
    st.write("🟥 FAKE" if prediction == 1 else "🟩 REAL")
