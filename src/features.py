import pandas
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline


def pipeline(transformers):
    return Pipeline([
        ("features", ColumnTransformer(list(transformers))),
        ("model", LogisticRegression())
    ])

def evaluate(name, model):
    scores = cross_validate(model, X_train, y_train, cv = cv_settings, scoring = ["accuracy", "f1_macro"])
    result = {
        "Experiment": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "AccuracyStd": scores["test_accuracy"].std(),
        "F1": scores["test_f1_macro"].mean(),
        "F1Std": scores["test_f1_macro"].std()
    }
    results.append(result)
    print(f"\t{name:45} accuracy {result["Accuracy"]:.3f} +- {result["AccuracyStd"]:.3f} | F1 macro {result["F1"]:.3f} +- {result["F1Std"]:.3f}")

pandas.set_option("display.width", 200)
pandas.set_option("display.max_columns", 20)

# 1. Loading and patterns
print("1. Loading and patterns:\n")

books = pandas.read_csv("../data/books_clean.csv")

books["HasColon"] = books["Title"].str.contains(":", regex = False)
books["HasSeries"] = books["Title"].str.contains(r"#\d", regex = True)

print(f"Loaded {len(books)} books")
print(books["Category"].value_counts().to_string())

# 2. Train / test split
print("\n2. Train / test split:\n")

X = books[["Title", "Description", "HasColon", "HasSeries"]]
y = books["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 0)

print(f"Training: {len(X_train)} books, test: {len(X_test)} books")
print(pandas.DataFrame({"Train": y_train.value_counts(), "Test": y_test.value_counts()}))

cv_settings = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 0)
results = []

# 3. Experiments
print("\n3. Experiments (5-fold cross-validation repeated 5 times, on the training books):\n")

PATTERNS = ("patterns", "passthrough", ["HasColon", "HasSeries"])

# Reference points
evaluate("1. Baseline (most frequent class)", DummyClassifier(strategy = "most_frequent"))
evaluate("2. Title patterns only", pipeline([PATTERNS]))

# Turning the description into numbers
evaluate("3. Description, word presence", pipeline([("description", CountVectorizer(binary = True), "Description")]))

evaluate("4. Presence, stop words removed", pipeline([("description", CountVectorizer(binary = True, stop_words = "english"), "Description")]))

for i in range(2, 6):
    evaluate(f"5.{i - 1} Presence, min_df = {i}", pipeline([("description", CountVectorizer(binary = True, min_df = i), "Description")]))

for i in range(2, 6):
    evaluate(f"6.{i - 1} Presence, no stop words, min_df = {i}", pipeline([("description", CountVectorizer(binary = True, min_df = i, stop_words = "english"), "Description")]))

# After manual evaluation, a description based on word presence with min_df = 4
# and stop words included will be used for further experiments
DESCRIPTION = ("description", CountVectorizer(binary = True, min_df = 4), "Description")

# Titles are short, so every title word is kept
TITLE = ("title", CountVectorizer(binary = True), "Title")

BIGRAMS = ("description", CountVectorizer(binary = True, min_df = 4, ngram_range = (1, 2)), "Description")

evaluate("7. Description + title", pipeline([DESCRIPTION, TITLE]))
evaluate("8. Description + title + patterns", pipeline([DESCRIPTION, TITLE, PATTERNS]))
evaluate("9. Description + title + patterns + bigrams", pipeline([BIGRAMS, TITLE, PATTERNS]))

# Results table
table = pandas.DataFrame(results).round(3)
print("\nResults:")
print(table.to_string())

table.to_csv("../data/feature_experiments.csv", index = False)
print("\nSaved table: feature_experiments.csv")
