# predict_sentiment.py
import joblib

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Get input tweet from the user
user_input = input("Enter a tweet to analyze: ")

# Transform the user's input using the vectorizer
user_input_vec = vectorizer.transform([user_input])

# Predict the sentiment using the trained model
prediction = model.predict(user_input_vec)

# Print the predicted sentiment
print(f"\nPredicted Sentiment: {prediction[0]}")
