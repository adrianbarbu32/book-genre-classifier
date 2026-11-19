import numpy
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def pipeline(model):
    return Pipeline([
        ("features", ColumnTransformer([DESCRIPTION, TITLE, PATTERNS])),
        ("model", model)
    ])

def evaluate(name, model):
    scores = cross_validate(model, X_train, y_train, cv = cv_settings, scoring = ["accuracy", "f1_macro"])
    result = {
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "AccuracyStd": scores["test_accuracy"].std(),
        "F1": scores["test_f1_macro"].mean(),
        "F1Std": scores["test_f1_macro"].std()
    }
    results.append(result)
    print(f"\t{name:45} accuracy {result["Accuracy"]:.3f} +- {result["AccuracyStd"]:.3f} | F1 macro {result["F1"]:.3f} +- {result["F1Std"]:.3f}")

pandas.set_option("display.width", 200)
pandas.set_option("display.max_columns", 20)

# 1. Loading and patterns (same as in features.py)
print("1. Loading and patterns:\n")

books = pandas.read_csv("../data/books_clean.csv")

books["HasColon"] = books["Title"].str.contains(":", regex = False)
books["HasSeries"] = books["Title"].str.contains(r"#\d", regex = True)

print(f"Loaded {len(books)} books")
print(books["Category"].value_counts().to_string())

# 2. Train / test split
# Same split as in features.py
print("\n2. Train / test split:\n")

X = books[["Title", "Description", "HasColon", "HasSeries"]]
y = books["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 0)

print(f"Training: {len(X_train)} books, test: {len(X_test)} books")
print(pandas.DataFrame({"Train": y_train.value_counts(), "Test": y_test.value_counts()}))

# Same cross-validation as in features.py
cv_settings = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 0)

# 3. Chosen features
print("\n3. Chosen features:\n")

# Chosen in features.py: experiment 8 (description + title + patterns), F1 macro 0.864 +- 0.054
# - word presence, stop words kept, min_df = 4 for the description
# - titles are short, so every title word is kept
DESCRIPTION = ("description", CountVectorizer(binary = True, min_df = 4), "Description")
TITLE = ("title", CountVectorizer(binary = True), "Title")
PATTERNS = ("patterns", "passthrough", ["HasColon", "HasSeries"])

# Number of features (the vocabularies are learned from the training books)
feature_transformer = ColumnTransformer([DESCRIPTION, TITLE, PATTERNS])
feature_transformer.fit(X_train)

for name, transformer in feature_transformer.named_transformers_.items():
    if hasattr(transformer, "vocabulary_"):
        print(f"{name}: {len(transformer.vocabulary_)} words")
    elif name == "patterns":
        print(f"patterns: {len(PATTERNS[2])} features")

feature_count = len(feature_transformer.get_feature_names_out())
print(f"Total: {feature_count} features for {len(X_train)} training books ({feature_count / len(X_train):.1f} features per book)")

# 4. Model selection and final evaluation
print("\n4. Model selection and final evaluation:\n")

results = []

# Reference point
evaluate("1. Baseline (most frequent class)", DummyClassifier(strategy = "most_frequent"))

# Logistic regression
evaluate("2. LogisticRegression", pipeline(LogisticRegression()))

# BernoulliNB
evaluate("3. BernoulliNB", pipeline(BernoulliNB()))

# LinearSVC
evaluate("4. LinearSVC", pipeline(LinearSVC()))

# Trees
evaluate("5. RandomForestClassifier", pipeline(RandomForestClassifier(random_state = 0)))

# Results table
table = pandas.DataFrame(results).round(3)
print("\nResults:")
print(table.to_string())

table.to_csv("../data/model_selection.csv", index = False)
print("\nSaved table: model_selection.csv")

best = table.loc[table["F1"].idxmax()]
print(f"\nBest: {best['Model']} (F1 macro {best['F1']:.3f} +- {best['F1Std']:.3f})")

# 5. Final training
print("\n5. Final training:\n")

# Chosen manually from the table above
CHOSEN_MODEL = "3. BernoulliNB"
final_model = pipeline(BernoulliNB())

final_model.fit(X_train, y_train)
print(f"Trained {CHOSEN_MODEL} on {len(X_train)} training books")

# 6. Final evaluation on the test set
print("\n6. Final evaluation on the test set:\n")

predictions = final_model.predict(X_test)
 
# Model scores
print(f"Chosen model: accuracy {accuracy_score(y_test, predictions):.3f}, "
      f"F1 macro {f1_score(y_test, predictions, average = 'macro'):.3f}\n")
print(classification_report(y_test, predictions, digits = 3))
 
# References on the same test books
dummy_predictions = DummyClassifier(strategy = "most_frequent").fit(X_train, y_train).predict(X_test)
rule_predictions = numpy.where(X_test["HasColon"], "Nonfiction", "Fiction")
 
references = pandas.DataFrame({
    "Model": ["Baseline (most frequent class)", "Rule \"colon = Nonfiction\"", CHOSEN_MODEL],
    "Accuracy": [accuracy_score(y_test, p) for p in [dummy_predictions, rule_predictions, predictions]],
    "F1": [f1_score(y_test, p, average = "macro") for p in [dummy_predictions, rule_predictions, predictions]]
}).round(3)
print("Comparison on the test books:")
print(references.to_string(index = False))
 
# Confusion matrix (image 2 in the report)
display = ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = "Blues")
display.ax_.set_title("Confusion matrix on the test books")
display.figure_.tight_layout()
display.figure_.savefig("../figures/confusion_matrix.png", dpi = 150)
print("\nSaved figure: confusion_matrix.png")
 
# Wrongly classified books
wrong = predictions != y_test.to_numpy()
print(f"\nWrongly classified books: {wrong.sum()} of {len(y_test)}")
for title, description, real, predicted in zip(X_test["Title"][wrong], X_test["Description"][wrong], y_test[wrong], predictions[wrong]):
    print(f"\t{title} | real: {real}, predicted: {predicted}")
    print(f"\t   {description[:150]}...")
 
# Most important features
model = final_model.named_steps["model"]
feature_names = final_model.named_steps["features"].get_feature_names_out()
print(f"\nClasses: {list(model.classes_)}")
 
# BernoulliNB: log P(feature present | Fiction) - log P(feature present | Nonfiction)
importance = model.feature_log_prob_[0] - model.feature_log_prob_[1]
 
weights = pandas.DataFrame({"Feature": feature_names, "Fiction score": importance}).round(3)
print("\nPushing most towards Fiction:")
print(weights.nlargest(15, "Fiction score").to_string(index = False))
print("\nPushing most towards Nonfiction:")
print(weights.nsmallest(15, "Fiction score").to_string(index = False))
 
# Interpretation: test score consistency with the cross-validation score
chosen_cv = table.loc[table["Model"] == CHOSEN_MODEL].iloc[0]
test_f1 = f1_score(y_test, predictions, average = "macro")

print(f"\nCross-validation F1 macro: {chosen_cv['F1']:.3f} +- {chosen_cv['F1Std']:.3f}")
print(f"Test F1 macro: {test_f1:.3f}")
print(f"One book is {1 / len(y_test):.1%} of the test accuracy")
