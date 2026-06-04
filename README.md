# Spam Message Detector

A machine learning web app that classifies text messages as **spam** or **ham** using **TF-IDF vectorization** and **Logistic Regression**. The project includes data cleaning, exploratory data analysis, model training, evaluation, and deployment with Streamlit.

## Project Overview

Spam messages often contain patterns such as promotional language, urgent wording, phone numbers, links, prize claims, money symbols, and excessive capitalization. This project analyzes those patterns and builds a supervised machine learning model to classify messages as spam or not spam.

The final model is deployed as an interactive Streamlit app where users can enter a message and receive a prediction with spam/ham probabilities.


## Dataset

The dataset contains SMS-style text messages labeled as either:

* `ham`: normal message
* `spam`: unwanted/promotional/scam-like message

Main columns:

| Column     | Description                    |
| ---------- | ------------------------------ |
| `Category` | Message label: `ham` or `spam` |
| `Message`  | Raw text message               |

## Exploratory Data Analysis

Before modeling, I performed EDA to understand how spam messages differ from ham messages.

Key findings:

| Metric                         |    Ham |   Spam |
| ------------------------------ | -----: | -----: |
| Message count                  |   4516 |    641 |
| Average character count        |  70.87 | 137.12 |
| Average word count             |  14.33 |  23.66 |
| Average exclamation count      |   0.18 |   0.69 |
| Average digit count            |   0.30 |  15.33 |
| Average money symbol count     |  0.005 |  0.412 |
| Average uppercase ratio        |  0.057 |  0.111 |
| URL rate                       | 0.0002 | 0.1388 |
| Phone-number-like pattern rate | 0.0007 | 0.7629 |

Main EDA conclusions:

* Spam messages are much longer than ham messages.
* Spam messages contain far more digits and phone-number-like patterns.
* Spam messages are more likely to contain URLs and money symbols.
* Spam messages use more uppercase letters and exclamation marks.
* These patterns suggest that text features and engineered features can be useful for spam classification.

## Model Approach

The project uses a scikit-learn pipeline:

```text
Raw message text → TF-IDF Vectorizer → Logistic Regression → Prediction
```

### Why TF-IDF?

TF-IDF converts raw text into numerical features by measuring how important words or phrases are in a message compared to the full dataset.

### Why Logistic Regression?

Logistic Regression is a strong baseline for text classification because it is:

* Fast
* Interpretable
* Effective on sparse TF-IDF features
* Easy to deploy

## Pipeline

The model pipeline combines preprocessing and classification into one object:

```python
spam_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])
```

Using a pipeline makes deployment easier because the app can pass raw user text directly into the model.

## Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

For spam detection, precision and recall are especially important:

* **Precision**: When the model predicts spam, how often is it correct?
* **Recall**: How many actual spam messages does the model successfully catch?

## App Features

The Streamlit app allows users to:

* Enter a custom message
* Predict whether the message is spam or ham
* View ham probability
* View spam probability

Example messages to test:

```text
Congratulations! You won a free prize. Call now!
```

```text
Hey, are we still meeting at 5 today?
```

## Project Structure

```text
spam-detector/
│
├── data/
│   └── spam.csv
│
├── app.py
├── train_model.py
├── emails.ipynb
├── spam_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone PASTE_YOUR_GITHUB_REPO_LINK_HERE
cd spam-detector
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python3 train_model.py
```

This creates the saved model file:

```text
spam_model.pkl
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Joblib
* Streamlit
* Git/GitHub


## What I Learned

Through this project, I practiced:

* Text classification
* Exploratory data analysis for NLP
* Handling duplicate text data
* TF-IDF feature extraction
* Training and evaluating classification models
* Building scikit-learn pipelines
* Saving and loading trained models
* Deploying a machine learning model with Streamlit
