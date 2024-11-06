import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
fake_news = pd.read_csv("fake.csv")
true_news = pd.read_csv("True.csv")

# Assigning  labels
fake_news['label'] = 0
true_news['label'] = 1

# Creating training and testing datasets
fake_train, fake_test = train_test_split(fake_news, test_size=0.2, random_state=42)
true_train, true_test = train_test_split(true_news, test_size=0.2, random_state=42)

# Combining the training and testing sets
train_set = pd.concat([fake_train, true_train])
test_set = pd.concat([fake_test, true_test])

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# Preprocess the text data
train_set['text'] = train_set['text'].apply(preprocess_text)
test_set['text'] = test_set['text'].apply(preprocess_text)

# Use TF-IDF for feature extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_set['text'])
y_train = train_set['label']
X_test = vectorizer.transform(test_set['text'])
y_test = test_set['label']

# Initialize the model with regularization
model = LogisticRegression(C=0.1, solver='liblinear')

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validated scores:", cv_scores)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
