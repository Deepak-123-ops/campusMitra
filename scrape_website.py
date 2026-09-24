import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

BASE_URL = "https://www.eatm.in/"

visited_urls = set()
all_text = []

headers = {
    "User-Agent": "CampusMitra-Research-Bot/1.0"
}


def scrape_page(url):

    if url in visited_urls:
        return

    visited_urls.add(url)

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        # Fix website encoding
        response.encoding = response.apparent_encoding

        if response.status_code != 200:
            return

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if text:

            all_text.append(
                f"\n\nSOURCE: {url}\n{text}"
            )

        print("Scraped:", url)

        # Find links
        for link in soup.find_all("a", href=True):

            next_url = urljoin(
                url,
                link["href"]
            )

            parsed_url = urlparse(next_url)

            # Stay within the same website
            if parsed_url.netloc == urlparse(BASE_URL).netloc:

                if next_url.startswith(BASE_URL):

                    if next_url not in visited_urls:

                        scrape_page(next_url)

    except Exception as error:

        print("Error:", url, error)


# Start scraping
scrape_page(BASE_URL)


# Create data folder
os.makedirs("data", exist_ok=True)


# Save collected text
with open(
    "data/college_data.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(all_text)
    )


print("\nScraping completed!")
print("Total pages:", len(visited_urls))
print("Saved to: data/college_data.txt")