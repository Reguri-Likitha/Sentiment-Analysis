# model_training.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('datasets/Tweets.csv')

# Use only relevant columns
df = df[['text', 'airline_sentiment']]

# Features and labels
X = df['text']
y = df['airline_sentiment']

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Create and train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')  # Save the model
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer

# Make predictions and check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

