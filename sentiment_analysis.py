import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('datasets/Tweets.csv')

# Check the first 5 rows of the dataset to confirm it loaded correctly
print(df.head())

# Check the column names to verify if the dataset has the expected structure
print(df.columns)

# Check if there are any missing values
print(df.isnull().sum())

# Check the distribution of the sentiment labels (positive, negative, neutral)
print(df['airline_sentiment'].value_counts())

