import spacy
from spacy.lang.en import English
from spacy import displacy
from spacy.pipeline import EntityRuler
import pandas as pd

# ==========================================
# 1. Load Dataset (CoNLL-2003 Mock)
# ==========================================
# In a real scenario, you would load 'train.txt' from the CoNLL dataset.
# Here we create a representative sample of that data format.

data = {
    'sentence': [
        "Apple Inc. is planning to open a new store in San Francisco.",
        "Elon Musk bought Twitter for $44 billion in 2022.",
        "The World Health Organization announced a new vaccine protocol.",
        "Michael Jordan played for the Chicago Bulls.",
        "I visited Paris and London last summer.",
        "Google's headquarters are in Mountain View, California.",
        "Amazon delivers packages to New York everyday."
    ]
}
df = pd.DataFrame(data)

# ==========================================
# 2. Rule-Based NER (The "Manual" Way)
# ==========================================
# We define explicit patterns to find entities. useful for custom/rare terms.

nlp_rules = English()
ruler = nlp_rules.add_pipe("entity_ruler")

# Define patterns
patterns = [
    {"label": "ORG", "pattern": "Apple Inc."},
    {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]},
    {"label": "PERSON", "pattern": [{"LOWER": "elon"}, {"LOWER": "musk"}]},
    {"label": "ORG", "pattern": "Twitter"},
    {"label": "MONEY", "pattern": [{"IS_CURRENCY": True}, {"LIKE_NUM": True}, {"LOWER": "billion"}]}
]

ruler.add_patterns(patterns)

print("--- Rule-Based Extraction ---")
doc1 = nlp_rules(df['sentence'].iloc[1]) # "Elon Musk bought Twitter..."
for ent in doc1.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
print("\n")


# ==========================================
# 3. Model-Based NER (The "AI" Way)
# ==========================================
# Using a pre-trained statistical model (en_core_web_sm)

nlp_model = spacy.load("en_core_web_sm")

print("--- Model-Based Extraction (en_core_web_sm) ---")
doc2 = nlp_model(df['sentence'].iloc[1])
for ent in doc2.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
print("\n")

# ==========================================
# BONUS 1: Visualization with displaCy
# ==========================================
# This usually renders in a browser or notebook. 
# We will render it to a simple block to show it works.

print("--- Visualizing Entities (Rendered HTML) ---")
html = displacy.render(doc2, style="ent", jupyter=False)
# In a Jupyter Notebook, you would just run: displacy.render(doc2, style="ent")
print("(Visualization generated successfully - run in Jupyter to see colors)\n")


# ==========================================
# BONUS 2: Model Comparison (Small vs Medium)
# ==========================================
# 'en_core_web_md' has word vectors, making it smarter at context than 'sm'.

try:
    nlp_md = spacy.load("en_core_web_md")
    
    test_sentence = "I saw Jaguar speeding down the highway."
    
    print(f"--- Comparing Models on Ambiguous Text: '{test_sentence}' ---")
    
    doc_sm = nlp_model(test_sentence)
    print("Small Model detected:", [(ent.text, ent.label_) for ent in doc_sm.ents])
    
    doc_md = nlp_md(test_sentence)
    print("Medium Model detected:", [(ent.text, ent.label_) for ent in doc_md.ents])
    
except OSError:
    print("Please download 'en_core_web_md' to run the comparison.")
    print("Run: python -m spacy download en_core_web_md")
