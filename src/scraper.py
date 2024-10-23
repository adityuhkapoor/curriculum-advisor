# scraper.py

import requests
from bs4 import BeautifulSoup
import html2text
import re
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

def scrape_and_save(url):
    html_content, safe_title = scrape_website(url)

    if html_content:
        markdown_text = convert_html_to_markdown(html_content)
        filename = f"{safe_title or 'untitled_site'}.mdx"
        filepath = os.path.join('data', 'curriculum', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        logging.info(f"Saved markdown for {url} as {filename}")
    else:
        logging.error(f"Failed to process {url}")

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Successfully fetched {url}")
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    html_content = soup.prettify()
    title_tag = soup.title
    title = title_tag.string if title_tag else "No Title"
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-')
    return html_content, safe_title

def convert_html_to_markdown(html_content):
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False  # Include links in the conversion
    markdown_content = markdown_converter.handle(html_content)
    return markdown_content

def main():
    urls = []

    with open('data/urls.txt', 'r') as file:
        for line in file:
            urls.append(line.strip())

    # Process URLs in parallel using ThreadPoolExecutor
    max_workers = min(10, len(urls))  # Limit the number of threads to avoid overwhelming the system
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scrape_and_save, url) for url in urls]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing URL: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
