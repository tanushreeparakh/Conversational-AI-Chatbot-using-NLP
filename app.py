import streamlit as st
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data
data = pd.read_csv("knowledge_base.csv")

questions = data['question'].tolist()
answers = data['answer'].tolist()

stop_words = set(stopwords.words('english'))

def preprocess(text):

    words = text.lower().split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match = similarity.argmax()

    score = similarity[0][best_match]

    if score > 0.2:
        return answers[best_match]

    return "Sorry, I couldn't understand your question."

# Streamlit UI

st.title("NLP Conversational AI Chatbot")

user_input = st.text_input(
    "Ask a question:"
)

if user_input:

    response = chatbot_response(user_input)

    st.write("### Bot Response")
    st.success(response)