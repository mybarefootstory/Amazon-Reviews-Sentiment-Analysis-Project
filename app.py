# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# 2. Define the input model
class WhatWeNeed(BaseModel):
    text: str  # Use 'str' instead of 'string'

# 3. Create the app object
app = FastAPI()

# Function to load a model from a pickle file
def load_model(file_name):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model

# Load models
le = load_model('sentiment_analysis_encoder.pkl')
tfidf = load_model('sentiment_analysis_vectorizer.pkl')
xgb_clf = load_model('sentiment_analysis_model.pkl')

@app.get('/')
def index():
    return {'message': 'Hello, Recruiters.!'}


@app.post('/predict')
def predict_sentiment(data: WhatWeNeed):
    # Preprocess the input text (transform it into TF-IDF form)
    text_tfidf = tfidf.transform([data.text])

    # Make prediction using the trained XGBoost model
    sentiment_encoded = xgb_clf.predict(text_tfidf)

    # Decode the predicted sentiment back to the original label
    sentiment_decoded = le.inverse_transform(sentiment_encoded)

    return {
        'prediction': sentiment_decoded[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


'''
1.
{
  "text": "I got this for my Mum who is not diabetic but needs to watch her sugar intake, and my father who simply chooses to limit unnecessary sugar intake - she's the one with the sweet tooth - they both LOVED these toffees, you would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free!  i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office so i'll eat them instead of snacking on sugary sweets.These are just EXCELLENT!"
}
{
  "prediction": "Positive"
}
2.
{
  "text": "The candy is just red , No flavor . Just  plan and chewy .  I would never buy them again"
}
{
  "prediction": "Negative"
}
3.
{
  "text": "This seems a little more wholesome than some of the supermarket brands, but it is somewhat mushy and doesn't have quite as much flavor either.  It didn't pass muster with my kids, so I probably won't buy it again."
}
{
  "prediction": "Neutral"
}
'''
