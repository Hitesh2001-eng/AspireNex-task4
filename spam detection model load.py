import joblib

# Load the model
loaded_model = joblib.load('spam_classifier_model.pkl')
loaded_tfidf = joblib.load('model.pkl')

# Example prediction
new_messages = ["Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now.",
                "Hey, are we still meeting for coffee tomorrow?"]
new_messages_tfidf = loaded_tfidf.transform(new_messages)
predictions = loaded_model.predict(new_messages_tfidf)

# Print predictions
print(f'Predictions: {predictions}')  # Output will be [1, 0],1 is spam and 0 is ham

# Optional: Mapping the prediction results back to labels
label_mapping = {0: 'ham', 1: 'spam'}
predicted_labels = [label_mapping[pred] for pred in predictions]
print(f'Predicted Labels: {predicted_labels}')
