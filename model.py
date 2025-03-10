import pandas as pd
import re 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


def preprocess_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)  
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\w+)', lambda m: m.group(1), text)
    text = text.replace("rn", "registered nurse")  
    return text.strip()


train['cleaned_text'] = train['text'].apply(preprocess_data)
test_data['cleaned_text'] = test_data['text'].apply(preprocess_data)


train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)  # Changed variable names

# Vectorization (fit only on training data)
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=15_000)
X_train = tfidf.fit_transform(train_df["cleaned_text"])
X_val = tfidf.transform(val_df["cleaned_text"])  
y_train = train_df["target"]
y_val = val_df["target"]  


model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)


val_preds = model.predict(X_val)

# Evaluate on VALIDATION (not test_data)
print("\nValidation Metrics:")
print(f"Accuracy: {accuracy_score(y_val, val_preds):.2f}")
print(f"Precision: {precision_score(y_val, val_preds):.2f}")
print(f"Recall: {recall_score(y_val, val_preds):.2f}")
print(f"F1-score: {f1_score(y_val, val_preds):.2f}")

print("--------------Deon you are a genius😂😂-----------------------")

X_competition_test = tfidf.transform(test_data["cleaned_text"])
test_probs = model.predict_proba(X_competition_test)[:, 1]

print(f"Test data shape: {X_competition_test.shape}")
print(f"Test data shape: {test_probs.shape}")
submission = pd.DataFrame({'ID': test_data['ID'], 'target': test_probs})
submission.to_csv('submission.csv', index=False)