import re

import ftfy
import pandas
from langdetect import DetectorFactory, detect

COLUMNS = ["Title", "Description", "Category"]
DESC_WORDS_LIMIT = 15

# Check if the description has a cut-off preview at the start, and remove it
def remove_duplicated_start(description, n = 50):
    chars = [c for c in description if not c.isspace()][: n]
    pattern = r"\s*".join(re.escape(c) for c in chars)
    match = re.compile(pattern).search(description, 1)

    if match:
        description = description[match.start() : ]
    return description

pandas.set_option("display.max_colwidth", 1000)
books = pandas.read_csv("../data/books.csv")
start_books_count = len(books)

# Initial book
print(f"Loaded {start_books_count} books")
print(books["Category"].value_counts().to_string())

# 1. Missing values
print("\n1. Missing values:")
for col in COLUMNS:
    count = books[col].isna().sum()
    books = books.dropna(subset = [col])
    print(f"\tRemoved {count} books without a {col}")

# 2. Encoding check for broken characters
print("\n2. Encoding check for broken characters:")
for col in COLUMNS:
    count = books[col].str.contains(r"Â|â€|Ã[\x80-\xbf]|\ufffd", regex = True, na = False).sum()
    books[col] = books[col].apply(ftfy.fix_encoding)
    print(f"\t{col}: fixed {count} values")

# 3. Text cleaning ("... more" ending + duplicated description preview)
print("\n3. Text cleaning:")
count = books["Description"].str.contains(r"\.\.\.\s*more\s*$", regex = True).sum()
books["Description"] = books["Description"].str.replace(r"\s*\.\.\.\s*more\s*$", "", regex = True)
print(f"\tStripped {count} description endings with \"... more\"")

# Uncomment to see if the amount of duplicated findings change across an interval
# for i in range(30, 200, 5):
#     changed = books["Description"] != books["Description"].apply(remove_duplicated_start, n = i)
#     print(i, changed.sum())

before = books["Description"].copy()
books["Description"] = books["Description"].apply(remove_duplicated_start)
count = (before != books["Description"]).sum()
print(f"\tRemoved {count} repeated description preview")

for col in COLUMNS:
    books[col] = books[col].apply(lambda t: " ".join(t.split()))

# The description either repeats the title or is too short for a real description
description_words = books["Description"].str.split().str.len()
removed_descriptions = (books["Description"].str.lower() == books["Title"].str.lower()) | (description_words < DESC_WORDS_LIMIT)

print(f"\tRemoved {removed_descriptions.sum()} books without a real description")
for i in books.index[removed_descriptions]:
    print(f"\t   {books.at[i, "Title"]} ({books.at[i, "Category"]}): {books.at[i, "Description"]!r}")
books = books[~removed_descriptions]

# 4. Duplicates
print("\n4. Duplicates:")
for col in ["Title", "Description"]:
    count = len(books)
    books = books.drop_duplicates(subset = [col])
    print(f"\tRemoved {count - len(books)} books with a duplicated {col}")

# 5. Language
print("\n5. Language:")
DetectorFactory.seed = 0
lang = books["Description"].apply(detect)
count = (lang != "en").sum()
for i in books.index[lang != "en"]:
    print(f"\t Removed {books.at[i, "Title"]} in {lang[i]} language")
print(f"\tRemoved {count} books which are not in English")
books = books[lang == "en"]

# Summary
books = books.reset_index(drop = True)
removed_books = start_books_count - len(books)

print(f"\nRemoved {removed_books} books in total")
print("Data Frame shape:", books.shape)
print(f"{books["Category"].value_counts().to_string()}")

books.to_csv("../data/books_clean.csv", index = False)
