from flask import Flask, render_template, request, jsonify
import pickle
import re
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load sample news data
sample_news_data = pd.read_csv('cleaned_combined_news.csv')[['text', 'Label']].values.tolist()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sample_news')
def sample_news():
    samples = random.sample(sample_news_data, 1)
    return render_template('sample_news.html', samples=samples)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['news_text']
    cleaned_input = preprocess_text(data)
    transformed_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(transformed_input)
    result = "Real" if prediction[0] == 1 else "Fake"  # Ensure this logic aligns with your model
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
