import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the datasets
train_data = pd.read_csv("train_data.csv")
validation_data = pd.read_csv("validation_data.csv")
test_data = pd.read_csv("test_data.csv")

# Assuming the last columns are the labels
X_train = train_data.drop(columns=['Emotion', 'Sentiment'])
y_train_emotion = train_data['Emotion']
y_train_sentiment = train_data['Sentiment']

X_validation = validation_data.drop(columns=['Emotion', 'Sentiment'])
y_validation_emotion = validation_data['Emotion']
y_validation_sentiment = validation_data['Sentiment']

X_test = test_data.drop(columns=['Emotion', 'Sentiment'])
y_test_emotion = test_data['Emotion']
y_test_sentiment = test_data['Sentiment']

# Train a model for emotion classification
emotion_model = RandomForestClassifier()
emotion_model.fit(X_train, y_train_emotion)

# Train a model for sentiment classification
sentiment_model = RandomForestClassifier()
sentiment_model.fit(X_train, y_train_sentiment)


# Test emotion model
emotion_test_score = emotion_model.score(X_test, y_test_emotion)
print(f"Emotion Model Test Accuracy: {emotion_test_score}")

# Test sentiment model
sentiment_test_score = sentiment_model.score(X_test, y_test_sentiment)
print(f"Sentiment Model Test Accuracy: {sentiment_test_score}")
