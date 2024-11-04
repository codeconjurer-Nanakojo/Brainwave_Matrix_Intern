# Fake News Detector

## Overview

This project is a Fake News Detector that leverages advanced machine learning algorithms to identify and flag false information. The model analyzes text to determine the credibility of news articles, helping users discern between real and fake news.

## Features

- **Real-time Analysis**: Instantly analyze news articles and identify potential false information.
- **Machine Learning Algorithms**: Utilizes state-of-the-art ML models for high accuracy in detecting fake news.
- **User Friendly Interface**: Ensures a seamless and intuitive user experience.

## Setup and Installation

1. **Clone the Repository**,create a virtual environment,install dependencies , run the flusk app :
   ```sh
   git clone https://github.com/codeconjurer-Nanakojo/Brainwave_Matrix_Intern.git
   cd Brainwave_Matrix_Intern

    python3 -m venv venv
    source venv/bin/activate

   pip install -r requirements.txt
    flask run
Usage
Analyze News: Navigate to the homepage and input news text to analyze its credibility.

Sample News: View sample news articles and their predicted labels (Real or Fake).

Copy Functionality: Easily copy generated news texts from the Sample News page.

File Structure
app.py: Main application file containing routes and logic.

templates/: Directory containing HTML templates.

static/: Directory containing static files like CSS and JavaScript.

trained_model.pkl: Pre-trained machine learning model.

tfidf_vectorizer.pkl: TF-IDF vectorizer.

cleaned_combined_news.csv: Dataset used for training the model.

Model Details
Training Data: The model was trained on a dataset comprising labeled news articles.

Algorithm: Utilizes a classification algorithm (e.g., Logistic Regression, SVM) with TF-IDF vectorization.

Performance: Achieved high accuracy in detecting fake news during testing.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Terms of Service

The terms of service will be outlined here soon. For now, please refer to the MIT License for general usage guidelines.


Contact
Nathaniel Justice Kojo Mensah Email: nathanieljusticemensah@gmail.com




