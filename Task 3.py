import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ==========================================
# 1. Load Dataset (Mock Data)
# ==========================================
# For the Kaggle dataset, you usually have two files: 'True.csv' and 'Fake.csv'
# Uncomment the lines below to use real files:
# true_df = pd.read_csv('True.csv')
# fake_df = pd.read_csv('Fake.csv')
# true_df['target'] = 'real'
# fake_df['target'] = 'fake'
# df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

# --- MOCK DATA GENERATION (for demo purposes) ---
data = {
    'title': [
        "Government passes new tax bill", 
        "Aliens land in New York City!", 
        "Study shows coffee is good for health", 
        "Celebrity reveals secret to immortality",
        "Stock market reaches all-time high",
        "Man eats 500 hot dogs in one minute"
    ] * 20,
    'text': [
        "The senate approved the legislation today after a long debate.",
        "Eyewitnesses claim to have seen a UFO hovering over Manhattan.",
        "Researchers found that antioxidants in coffee reduce heart risk.",
        "The actor claims drinking magic water keeps him young forever.",
        "Investors are celebrating as the Dow Jones surged 500 points.",
        "Doctors are baffled by the man's ability to consume so much food."
    ] * 20,
    'target': ['real', 'fake', 'real', 'fake', 'real', 'fake'] * 20
}
df = pd.DataFrame(data)

# ==========================================
# 2. Preprocessing
# ==========================================
# Combine Title and Text for better context
df['content'] = df['title'] + " " + df['text']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    words = text.split()
    # Lemmatization: converts 'running' -> 'run', 'better' -> 'good'
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_content'] = df['content'].apply(clean_text)

# ==========================================
# 3. Vectorization (TF-IDF)
# ==========================================
# We use n_gram_range=(1,2) to capture phrases like "tax bill" or "white house"
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_content'])
y = df['target'].map({'real': 1, 'fake': 0}) # Encode labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Model Training (SVM)
# ==========================================
# LinearSVC is often faster and better for text than standard SVC
model = LinearSVC(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# 5. Evaluation
# ==========================================
print("--- Model Performance (SVM) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ==========================================
# BONUS: Word Cloud Visualization
# ==========================================
# Separate text by category
real_text = " ".join(df[df['target'] == 'real']['clean_content'])
fake_text = " ".join(df[df['target'] == 'fake']['clean_content'])

# Generate Word Clouds
wc_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
wc_fake = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(fake_text)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(wc_real, interpolation='bilinear')
plt.title("Most Frequent Words in REAL News")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wc_fake, interpolation='bilinear')
plt.title("Most Frequent Words in FAKE News")
plt.axis('off')

plt.tight_layout()
plt.show()