from urllib.parse import urljoin

import pandas
import requests
from bs4 import BeautifulSoup

FICTION_URL = "https://books.toscrape.com/catalogue/category/books/fiction_10/index.html"
NONFICTION_URL = "https://books.toscrape.com/catalogue/category/books/nonfiction_13/index.html"

# Scrapes every book of one category, page by page
def scrape_category(url, name) -> pandas.DataFrame:
    books = []
    page_counter = 1

    while url:
        soup = BeautifulSoup(requests.get(url, timeout = 10).content, "html.parser")

        book_links = [urljoin(url, book["href"]) for book in soup.select("h3 > a")]

        print(f"{name}: page {page_counter}, {len(book_links)} books")

        for book in book_links:
            book_soup = BeautifulSoup(requests.get(book, timeout = 10).content, "html.parser")

            title = book_soup.select_one("h1").text.strip()
            description = book_soup.select_one("#product_description").find_next_sibling().text.strip()
            category = book_soup.select("ul.breadcrumb a")[2].text
            image = urljoin(book, book_soup.select_one("#product_gallery img")["src"])

            books.append({"Title": title, "Description": description, "Image": image, "Category": category})

        next_link = soup.select_one("li.next a")
        if next_link:
            url = urljoin(url, next_link["href"])
        else:
            url = None

        page_counter += 1

    return pandas.DataFrame(books)    

books = pandas.concat([scrape_category(FICTION_URL, "Fiction"), scrape_category(NONFICTION_URL, "Nonfiction")], ignore_index = True)

print("Data Frame shape:", books.shape)
print(books["Category"].value_counts())

books.to_csv("../data/books.csv", index = False)
print("\nSaved file: books.csv")
