import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the CSV files into DataFrames
fake_df = pd.read_csv('fake.csv')
true_df = pd.read_csv('true.csv')

# Add a label column to each
fake_df['Label'] = 0
true_df['Label'] = 1

# Combine the DataFrames
df = pd.concat([fake_df, true_df])

# Shuffle the combined DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Check for missing values and drop them
df.dropna(subset=['text'], inplace=True)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])
    return text

# Apply text preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Save the cleaned combined data
df.to_csv('cleaned_combined_news.csv', index=False)
