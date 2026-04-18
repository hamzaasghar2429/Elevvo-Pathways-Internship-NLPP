import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import pyLDAvis.gensim_models
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# ==========================================
# 1. Load Dataset (BBC News Mock)
# ==========================================
# In reality: df = pd.read_csv('bbc_news.csv')
data = {
    'text': [
        # Tech / Business
        "Google announces new AI features for search engine.",
        "Apple stocks rise after new iPhone release.",
        "Microsoft acquires gaming giant Activision.",
        "Tech companies face new regulations in Europe.",
        "Amazon expands drone delivery service.",
        # Sports
        "Manchester United wins the championship league match.",
        "Tennis star breaks world record at Wimbledon.",
        "Olympic athlete tests positive for banned substance.",
        "Football team manager resigns after poor season.",
        "NBA finals reach record viewership numbers.",
        # Politics
        "Prime Minister announces new budget plan for education.",
        "Election results show a close race between candidates.",
        "Parliament votes on new environmental policy bill.",
        "Senator proposes tax cuts for small businesses.",
        "Global leaders meet to discuss climate change treaty."
    ] * 5 # Duplicate to give the model enough data
}
df = pd.DataFrame(data)

# ==========================================
# 2. Preprocessing
# ==========================================
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split()
    # Remove stopwords and short words
    return [word for word in tokens if word not in stop_words and len(word) > 2]

# Apply preprocessing
processed_docs = df['text'].map(preprocess)

# Create Dictionary (Map words to IDs)
dictionary = corpora.Dictionary(processed_docs)

# Create Corpus (Bag of Words frequency count)
corpus = [dictionary.doc2bow(text) for text in processed_docs]

# ==========================================
# 3. LDA Model (Gensim)
# ==========================================
print("--- Training LDA Model ---")
# num_topics=3 because we know we have Tech, Sports, Politics
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=3,
                                       passes=10,
                                       random_state=42)

# Print Topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")

# ==========================================
# BONUS 1: NMF Comparison (Scikit-Learn)
# ==========================================
print("\n--- Training NMF Model (Bonus) ---")

# NMF requires TF-IDF input, not just Bag of Words
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

nmf_model = NMF(n_components=3, random_state=42, init='nndsvd')
nmf_topics = nmf_model.fit_transform(tfidf)

# Function to display NMF topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(nmf_model, feature_names, 5)

# ==========================================
# BONUS 2: Visualization (pyLDAvis & WordCloud)
# ==========================================

# A. Word Cloud for Topic 0 (LDA)
plt.figure(figsize=(10,5))
# Get weights for Topic 0
topic0_terms = dict(lda_model.show_topic(0, 30)) 
wc = WordCloud(background_color="white", max_words=20).generate_from_frequencies(topic0_terms)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for LDA Topic 0")
plt.show()

# B. pyLDAvis (Interactive)
# NOTE: This line renders an interactive HTML file. 
# It works best in Jupyter Notebooks.
try:
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    # pyLDAvis.save_html(vis, 'lda_visualization.html') # Save to file
    print("\npyLDAvis preparation successful. Run 'pyLDAvis.display(vis)' in Jupyter to see it.")
except Exception as e:
    print(f"Visualization skipped: {e}")