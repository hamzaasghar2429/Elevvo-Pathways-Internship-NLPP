import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  # For Bonus
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)

# ==========================================
# 1. Load Dataset (Mock Data for Demo)
# ==========================================
# REPLACE THIS BLOCK with: df = pd.read_csv('IMDB Dataset.csv')
data = {
    'review': [
        "I loved this movie, it was fantastic and thrilling.",
        "Worst movie ever. Complete waste of time and money.",
        "The plot was boring and the acting was terrible.",
        "Amazing performance by the lead actor! A masterpiece.",
        "It was okay, average at best. Not great.",
        "Absolutely brilliant! I cried at the end.",
        "Disgusting and offensive content. Hated it.",
        "A beautiful story about love and loss.",
        "Script was weak and directionless.",
        "Highly recommended for everyone to watch!"
    ] * 10, # Duplicating to make dataset larger for split
    'sentiment': ['positive', 'negative', 'negative', 'positive', 'neutral', 
                  'positive', 'negative', 'positive', 'negative', 'positive'] * 10
}
df = pd.DataFrame(data)

# Filter out neutral reviews for Binary Classification
df = df[df['sentiment'] != 'neutral']

# ==========================================
# 2. Preprocessing
# ==========================================
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = ' '.join([w for w in text.split() if w not in stop_words]) # Remove stopwords
    return text

df['clean_text'] = df['review'].apply(preprocess_text)

# ==========================================
# 3. Vectorization (TF-IDF)
# ==========================================
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Model Training & Comparison (Bonus)
# ==========================================

# Model A: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Model B: Naive Bayes (Bonus Task)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

print(f"--- Model Comparison ---")
print(f"Logistic Regression Accuracy: {lr_acc:.2f}")
print(f"Naive Bayes Accuracy:       {nb_acc:.2f}")
print("\n")

# ==========================================
# 5. Visualize Top Words (Bonus)
# ==========================================
# We can use the Logistic Regression coefficients to find the most influential words
feature_names = tfidf.get_feature_names_out()
coefficients = lr_model.coef_[0]

# Create a dataframe of words and their weights
word_weights = pd.DataFrame({'word': feature_names, 'weight': coefficients})

# Top 5 Positive Words (Highest positive weight)
top_positive = word_weights.sort_values(by='weight', ascending=False).head(5)
# Top 5 Negative Words (Lowest negative weight)
top_negative = word_weights.sort_values(by='weight', ascending=True).head(5)

print("--- Bonus: Top Predictive Words ---")
print("Top Positive Words:\n", top_positive)
print("\nTop Negative Words:\n", top_negative)

# Simple Bar Plot for Visualization
plt.figure(figsize=(10, 5))
colors = ['green'] * 5 + ['red'] * 5
combined = pd.concat([top_positive, top_negative])
plt.bar(combined['word'], combined['weight'], color=colors)
plt.title("Top Positive (Green) vs Negative (Red) Words")
plt.xticks(rotation=45)
plt.show()
