# Amazon Reviews Sentiment Analysis Project

## Overview

In this project, I worked on building a sentiment analysis model using Amazon review data. I trained the model, deployed it as a microservice, and exposed APIs that can predict the sentiment (positive, neutral, or negative) of input text. Throughout the project, I followed the MVC architecture and implemented the service in a microservice format.

## Project Structure (MVC Pattern)

I structured the project around the MVC (Model-View-Controller) pattern to keep things organized:

- **Model**: The sentiment analysis model I trained on Amazon review data. It classifies reviews as positive, neutral, or negative. After training, I saved this model using `pickle`.
- **View**: The view is represented by the FastAPI microservice, which exposes RESTful endpoints to interact with the trained model for predictions.
- **Controller**: The FastAPI app controls the logic—handling requests, performing text preprocessing, loading the model, and returning predictions.

## Microservice Architecture

The project is deployed as a microservice, exposing RESTful APIs that allow users to interact with my sentiment analysis model. I used FastAPI to create the service, ensuring it is lightweight and easy to use.

### Endpoints:

- **GET /**: A simple welcome message from the service.
- **POST /predict**: Receives a JSON object with the text input, then returns the predicted sentiment (positive, neutral, or negative).
- **GET /health**: A basic health check endpoint to make sure the service is running properly (returns "running" if it's up).

## Steps to Run the Project

### 1. Data Collection

I started by loading the Amazon reviews dataset. To keep the processing manageable, I worked with 10,000 rows from the dataset.

```python
df = pd.read_csv('/content/data/amazon_reviews.csv', nrows=10000)
```

### 2. Data Exploration and Preprocessing

I cleaned the dataset by removing unnecessary columns like user and product IDs, and dropped any duplicates or null values to ensure the data was ready for analysis.

```python
df_cleaned = df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'Time'], axis=1)
df_cleaned = df_cleaned.dropna().drop_duplicates()
```

### 3. Text Preprocessing

The text data needed to be cleaned and prepared for model training. I performed steps like lowercasing, removing special characters, tokenizing, and lemmatizing. This helped to ensure that the text was in a suitable format for analysis.

```python
def clean_text(text):
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df_cleaned['cleaned_text'] = df_cleaned['Text'].apply(clean_text)
```

### 4. Sentiment Labeling

Based on the review scores, I labeled each review as either positive, neutral, or negative. Reviews with scores of 1 or 2 were labeled as negative, 3 as neutral, and 4 or 5 as positive.

```python
def label_sentiment(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

df_cleaned['sentiment'] = df_cleaned['Score'].apply(label_sentiment)
```

### 5. Model Training

I transformed the cleaned text into numerical form using TF-IDF vectorization and then trained an XGBoost classifier on the transformed data. I evaluated the model using accuracy scores, classification reports, and confusion matrices.

```python
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)

xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X_train_tfidf, y_train_encoded)
```

### 6. Model and Component Serialization

Once I had trained the model, I saved it (along with the TF-IDF vectorizer and the label encoder) to disk using the `pickle` module, so that I could use it later when deploying the microservice.

```python
with open('sentiment_analysis_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_clf, model_file)
```

### 7. API Deployment (FastAPI)

I built a RESTful API using FastAPI that allows users to send text reviews and get a sentiment prediction. Additionally, I included a health check endpoint to verify the service status.

```python
from fastapi import FastAPI

app = FastAPI()

@app.post('/predict')
def predict_sentiment(data: WhatWeNeed):
    text_tfidf = tfidf.transform([data.text])
    sentiment_encoded = xgb_clf.predict(text_tfidf)
    sentiment_decoded = le.inverse_transform(sentiment_encoded)
    return {'prediction': sentiment_decoded[0]}

@app.get('/health')
def health_check():
    return {'status': 'running'}
```

### 8. Running the Application

I deployed the application locally using Uvicorn, which runs the FastAPI app and serves it on a local host.

```bash
uvicorn.run(app, host='127.0.0.1', port=8000)
```

## Additional Features (Bonus)

As an additional task, I can integrate an external sentiment analysis API (such as OpenAI's GPT model or Hugging Face's sentiment model) and compare the results with my model. This can be done by adding a new `POST /predict_external` endpoint to the service.

## Conclusion

Through this project, I successfully built and deployed a sentiment analysis model using the MVC architecture and microservice principles. It handles the entire process—from data collection and model training to deployment and real-time predictions—making it a complete end-to-end solution.

---
