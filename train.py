import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts, labels = [], []
with open("spam_dataset.csv", newline="") as f:
    for row in csv.DictReader(f):
        texts.append(row["text"])
        labels.append(row["label"])

pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipe.fit(texts, labels)
joblib.dump(pipe, "model.joblib")
