import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:/Users/Adnan Fatawi/Documents/Dokumen Dataset PA/MELD.Raw.tar/MELD.Raw/MELD.Raw/test_sent_emo.csv")

df = df.drop(["Sr No.", "Speaker", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"], axis=1)

# ------------------------------------------ DATA CLEANING MISSING VALUE
# Count missing values in each column
missing_data = df.isnull().sum()
print("Missing data in each column:\n", missing_data)

# Count the total number of rows with missing data
rows_with_missing_data = df.isnull().any(axis=1).sum()
print(f"\nTotal rows with missing data: {rows_with_missing_data}")

# Drop rows with any missing values
df_cleaned = df.dropna()
print("\nDataFrame after dropping rows with missing data:\n", df_cleaned)

# ------------------------------------------ DATA CLEANING DUPLICATED VALUE
# Remove duplicates based only on the 'utterance' column
df = df.drop_duplicates(subset=['Utterance'], keep='first')

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_dataset_no_duplicates.csv', index=False)

print("\nDataFrame after dropping rows with duplicated data:\n", df)

# ------------------------------------------ DATA ENCODING CATEGORICAL VARIABLES
df = pd.read_csv('cleaned_dataset_no_duplicates.csv')

# Perform One-Hot Encoding on 'emotion' and 'sentiment' columns
df_encoded = pd.get_dummies(df, columns=['Emotion', 'Sentiment'])

df_encoded.to_csv('encoding_categorical.csv', index=False)

# Display the first few rows of the encoded DataFrame
print(df_encoded.head())

# ------------------------------------------ DATA FEATURE ENGINEERING
# 1. Feature Engineering for Emotion Classification
# Creating a new DataFrame with only 'utterance' and 'emotion'
emotion_classification_df = df[['Utterance', 'Emotion']]

# 2. Feature Engineering for Sentiment Classification
# Creating a new DataFrame with only 'utterance' and 'sentiment'
sentiment_classification_df = df[['Utterance', 'Sentiment']]

# Save the new DataFrames to CSV files
emotion_classification_df.to_csv('emotion_classification.csv', index=False)
sentiment_classification_df.to_csv('sentiment_classification.csv', index=False)

print("Feature engineering completed and files saved.")

# ------------------------------------------ TF-IDF VECTORIZATION
# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform the text data from the 'Utterance' column
X_tfidf = tfidf_vectorizer.fit_transform(df['Utterance'])

# Convert the TF-IDF matrix to a DataFrame
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# ------------------------------------------ DATA TRAIN, VALIDATION, TESTING

# Load your full dataset
# df = pd.read_csv('cleaned_dataset_no_duplicates.csv')

# # Define the split ratios
# train_ratio = 0.7
# validation_ratio = 0.15
# test_ratio = 0.15

# # First, split into training and temp datasets
# train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), random_state=42)

# # Then split the temp dataset into validation and test datasets
# validation_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (test_ratio + validation_ratio)), random_state=42)

# # Save the splits to separate CSV files
# train_data.to_csv("train_data.csv", index=False)
# validation_data.to_csv("validation_data.csv", index=False)
# test_data.to_csv("test_data.csv", index=False)

# print("Datasets successfully split and saved!")

# Test
# ------------------------------------------ SPLIT THE DATASET
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf_df, df['Emotion'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ------------------------------------------ TRAIN THE MODEL
# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
validation_accuracy = model.score(X_val, y_val)
test_accuracy = model.score(X_test, y_test)
print(f"Validation Accuracy: {validation_accuracy}")
print(f"Test Accuracy: {test_accuracy}")