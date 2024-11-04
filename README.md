
# Fake News Classifier Project

This project is focused on detecting and classifying fake news using multiple machine learning models. Leveraging various natural language processing (NLP) techniques and supervised learning algorithms, this classifier identifies fake news articles with high accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Models Implemented](#models-implemented)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation and Results](#evaluation-and-results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
With the rise of digital media, the spread of misinformation has become a significant issue. This project aims to create a robust model to classify news as real or fake. Multiple machine learning models, including Naive Bayes, SVM, Decision Tree, and others, have been trained and compared for this task.

## Dataset
The dataset used for training and evaluation contains labeled news articles. It includes fields such as the article's title, content, and its classification label (real or fake).

- **Features Used**: Text data from article titles and content
- **Labels**: Binary classification (Real, Fake)

## Installation
Ensure you have Python installed (version 3.7 or higher is recommended). Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Requirements
The main libraries used in this project include:
- `nltk`
- `autocorrect`
- `wordcloud`
- `tensorflow`
- `sklearn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## Models Implemented
The following machine learning models were trained and evaluated:
1. **Naive Bayes**
2. **Support Vector Machine (SVM)**
3. **Decision Tree**
4. **Logistic Regression**
5. **Passive Aggressive Classifier**
6. **AdaBoost**
7. **Gradient Boosting**
8. **Multilayer Perceptron (MLP)**

## Data Preprocessing
1. **Tokenization**: Tokenizing text into words and sentences.
2. **Stopwords Removal**: Common stop words were removed.
3. **Stemming and Lemmatization**: Normalizing text for consistency.
4. **Spelling Correction**: Using `autocorrect` for correcting common typos.
5. **Vectorization**: Transforming text into numerical representations using:
   - TF-IDF Vectorizer
   - Count Vectorizer
   - Hashing Vectorizer

## Model Training
Each model was trained using a portion of the data, while the rest was used for validation. The training and validation data splits were managed to prevent data leakage and ensure a fair evaluation of model performance.

## Evaluation and Results
The performance of each model was evaluated using standard metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Performance metrics are documented for each model to allow for comparison and selection of the best-performing classifier.

## Visualization
Data visualizations include:
- Word Clouds of the most common words in fake vs. real news.
- Performance comparison graphs for different models.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request if you have any improvements.

## License
This project is licensed under the MIT License.
