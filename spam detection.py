import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
file_path = "C:/Users/hites/Downloads/SPAM SMS DETECTION dataset/spam.csv"
df = pd.read_csv(file_path, encoding='latin-1')

# Drop irrelevant columns and rename the remaining ones
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocess the labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Transform the text data using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a classifier (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Example usage: Load the model and vectorizer
loaded_model = joblib.load('spam_classifier_model.pkl')
loaded_tfidf = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
new_messages = ["Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now.",
                "Hey, are we still meeting for coffee tomorrow?"]
new_messages_tfidf = loaded_tfidf.transform(new_messages)
predictions = loaded_model.predict(new_messages_tfidf)

print(f'Predictions: {predictions}')  # Output will be [1, 0], indicating the first message is spam and the second is ham


joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(tfidf, 'model.pkl')
