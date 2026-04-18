import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==========================================
# 1. Load Dataset (AG News)
# ==========================================
# For this demo, I'll create a dummy dataset. 
# REPLACE with: df = pd.read_csv('train.csv') (AG News usually has 'Title', 'Description', 'Class Index')

data = {
    'description': [
        "The stock market crashed today due to inflation fears.",
        "The team scored a goal in the final minute to win the championship.",
        "New AI model released by tech giant solves complex physics problems.",
        "President announces new trade deal with neighboring countries.",
        "Oil prices surge as OPEC cuts production.",
        "Olympic athlete breaks world record in 100m sprint.",
        "Smartphone sales drop as market becomes saturated.",
        "Peace treaty signed to end the long-standing conflict."
    ] * 20,
    'label': [3, 2, 4, 1, 3, 2, 4, 1] * 20 
    # AG News Labels: 1-World, 2-Sports, 3-Business, 4-Sci/Tech
}
df = pd.DataFrame(data)

# Map numeric labels to names for clarity
label_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
df['category'] = df['label'].map(label_map)

# ==========================================
# 2. Preprocessing (Lemmatization included)
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    words = text.split()
    # Remove stopwords AND Lemmatize (convert 'running' -> 'run')
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['description'].apply(clean_text)

# ==========================================
# 3. Vectorization & Split
# ==========================================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Model Training (Logistic Regression)
# ==========================================
model = LogisticRegression(multi_class='ovr', solver='lbfgs') # 'ovr' = One-vs-Rest
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=label_map.values()))

# ==========================================
# BONUS 1: Visualization (Word Cloud)
# ==========================================
# Generate a Word Cloud for the 'Sports' category
sports_text = " ".join(df[df['category'] == 'Sports']['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sports_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Sports Category")
plt.show()

# ==========================================
# BONUS 2: Neural Network (Keras)
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Prepare targets for NN (One-Hot Encoding: 1 -> [1,0,0,0])
# Note: AG News labels are 1-4, Keras expects 0-3 usually, so we subtract 1
y_train_nn = to_categorical(y_train - 1, num_classes=4)
y_test_nn = to_categorical(y_test - 1, num_classes=4)

# Simple Feedforward Neural Network
nn_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'), # Input Layer
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(32, activation='relu'), # Hidden Layer
    Dense(4, activation='softmax') # Output Layer (4 classes)
])

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n--- Training Neural Network ---")
history = nn_model.fit(X_train, y_train_nn, epochs=5, batch_size=16, verbose=1, validation_data=(X_test, y_test_nn))
