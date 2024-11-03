import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Function to preprocess input text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters, numbers, etc.
    text = text.lower()  # Convert to lowercase
    return text


def predict_news():
    user_input = input("Enter news text: ")

    # Preprocess the input
    cleaned_input = preprocess_text(user_input)

    # Transform the input using the TF-IDF Vectorizer
    transformed_input = vectorizer.transform([cleaned_input])

    # Predict using the trained model
    prediction = model.predict(transformed_input)

    # Output the prediction
    result = "Real" if prediction[0] == 0 else "Fake"
    print(f"The news is predicted to be: {result}")


# Call the function
if __name__ == "__main__":
    predict_news()