from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('../cleaned_combined_news.csv')

# Fill NaN values in the 'text' column with an empty string and ensure all values are strings
data['text'] = data['text'].fillna('').astype(str)

# Function to generate word cloud
def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Generate word clouds
generate_wordcloud(data[data['Label'] == 0]['text'], 'Word Cloud - Fake News', 'wordcloud_fake.png')
generate_wordcloud(data[data['Label'] == 1]['text'], 'Word Cloud - Real News', 'wordcloud_real.png')
