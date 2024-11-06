import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('../cleaned_combined_news.csv')

# Data distribution
plt.figure(figsize=(6, 6))
data['Label'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribution of Real and Fake News')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Fake', 'Real'])
plt.savefig('distribution.png')
plt.close()
