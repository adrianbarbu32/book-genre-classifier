import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

matplotlib.use("Agg")

pandas.set_option("display.width", 200)
pandas.set_option("display.max_columns", 20)

# 1. Loading and types
print("1. Loading and types\n")

books = pandas.read_csv("../data/books_clean.csv")
books["Category"] = books["Category"].astype("category")
books["IsFiction"] = (books["Category"] == "Fiction").astype(int)

books.info()

# 2. Classes
print("\n2. Classes\n")

print(books["Category"].value_counts().to_string(), "\n")
shares = books["Category"].value_counts(normalize = True)
print(shares.round(3).to_string())
print(f"\nBaseline (always predict '{shares.idxmax()}'): {shares.max():.1%}")

# 3. Lengths
print("\n3. Lengths\n")

books["TitleLength"] = books["Title"].str.split().str.len()
books["DescriptionLength"] = books["Description"].str.split().str.len()

for col in ["TitleLength", "DescriptionLength"]:
    print(f"\n{col}:")
    print(books.groupby("Category")[col].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize = (10, 4))
books.boxplot(column = "TitleLength", by = "Category", ax = axes[0])
axes[0].set_title("Title length")
axes[0].set_ylabel("words")
books.boxplot(column = "DescriptionLength", by = "Category", ax = axes[1])
axes[1].set_title("Description length")
axes[1].set_ylabel("words")
fig.suptitle("")
fig.tight_layout()
fig.savefig("../figures/lengths_boxplot.png", dpi=150)
print("\nSaved figure: lengths_boxplot.png")

# 4. Title patterns
print("\n4. Title patterns")

books["HasColon"] = books["Title"].str.contains(":", regex = False)
books["HasSeries"] = books["Title"].str.contains(r"#\d", regex = True)

for col in ["HasColon", "HasSeries"]:
    print(f"\n{col}:")
    print(pandas.crosstab(books["Category"], books[col], margins = True))

# Rule: colon in title = Nonfiction
prediction = numpy.where(books["HasColon"], "Nonfiction", "Fiction")
correct = prediction == books["Category"]
print(f"Rule \"colon = Nonfiction, else Fiction\": accuracy {correct.mean():.1%}")

print(f"\nTitles where the rule is wrong: {(~correct).sum()}")
for title, category in books.loc[~correct, ["Title", "Category"]].values:
    print(f"\t[{category}] {title}")

# 5. Correlation
print("\n5. Correlation\n")

corr = books[["TitleLength", "DescriptionLength", "HasColon", "HasSeries", "IsFiction"]].corr()
print(corr.round(2))
 
print("\nCorrelation with IsFiction:")
print(corr["IsFiction"].drop("IsFiction").round(2).to_string())
print(f"\nHasColon vs TitleLength (possible redundancy): {corr.loc['HasColon', 'TitleLength']:.2f}")

# Title length correlation when there is no colon
no_colon = books[~books["HasColon"]]
print(f"\nTitles without a colon: {len(no_colon)}")
print(no_colon.groupby("Category")["TitleLength"].describe().round(1))
no_colon_corr = no_colon[["TitleLength", "IsFiction"]].corr().loc["TitleLength", "IsFiction"]
print(f"TitleLength vs IsFiction, titles without a colon: {no_colon_corr:.2f}")

# 6. Words in descriptions
print("\n6. Words in descriptions\n")

vectorizer = CountVectorizer(binary = True)
words_matrix = vectorizer.fit_transform(books["Description"])
vocabulary = vectorizer.get_feature_names_out()

description_freq = numpy.asarray(words_matrix.sum(axis = 0)).ravel()
print(f"Vocabulary size: {len(vocabulary)} words")
print(f"Words present in only one description: {(description_freq == 1).sum()} ({(description_freq == 1).mean():.1%})")

# Per class analysis
fiction_mask = (books["IsFiction"] == 1).to_numpy()
fiction_count = fiction_mask.sum()
nonfiction_count = (~fiction_mask).sum()

words = pandas.DataFrame({
    "Word": vocabulary,
    "Fiction": numpy.asarray(words_matrix[fiction_mask].sum(axis = 0)).ravel(),
    "Nonfiction": numpy.asarray(words_matrix[~fiction_mask].sum(axis = 0)).ravel()
})
words["FictionPercentage"] = (words["Fiction"] / fiction_count * 100).round(1)
words["NonfictionPercentage"] = (words["Nonfiction"] / nonfiction_count * 100).round(1)
words["Difference"] = words["FictionPercentage"] - words["NonfictionPercentage"]

# Biggest difference between classes, with appearances in at least 5 books
common_words = words[words["Fiction"] + words["Nonfiction"] >= 5]
print(f"\nWords in at least 5 books: {len(common_words)}")
print("\nMost typical for Fiction:")
print(common_words.nlargest(15, "Difference").to_string(index = False))
print("\nMost typical for Nonfiction:")
print(common_words.nsmallest(15, "Difference").to_string(index = False))

# Most frequent words per class, without stop words
stop_words = set(ENGLISH_STOP_WORDS)

freq_words = words[~words["Word"].isin(stop_words)]
print("\nMost frequent in Fiction (without stop words):")
print(freq_words.nlargest(15, "Fiction").to_string(index = False))
print("\nMost frequent in Nonfiction (without stop words):")
print(freq_words.nlargest(15, "Nonfiction").to_string(index = False))

pronouns = ["you", "your", "we", "he", "his", "she", "her", "they"]
print("\nPronouns:")
print(words.set_index("Word").reindex(pronouns))
