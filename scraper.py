import requests
from bs4 import BeautifulSoup
import html2text
import re

def scrape_website(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        html_content = soup.prettify()
        title_tag = soup.title
        title = title_tag.string if title_tag else "No Title"
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-')
        return html_content, safe_title
    else:
        print("failed to return")
        return None, None

def convert_html_to_markdown(html_content):
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False  # Include links in the conversion
    markdown_content = markdown_converter.handle(html_content)
    return markdown_content

urls = []

with open('urls.txt', 'r') as file:
    for line in file:
        # Strip any extra whitespace (like newline characters) and append to the list
        urls.append(line.strip())
# Example usage
for url in urls:
    html_content, safe_title = scrape_website(url)

    if html_content:
        markdown_text = convert_html_to_markdown(html_content)
        if safe_title:
            with open(f'curriculumData/{safe_title}.mdx', 'w') as f:
                f.write(markdown_text)
        else:
            with open(f'curriculumData/untitled_site.mdx', 'w') as f:
                f.write(markdown_text)

        print(markdown_text)
    else:
        print("failed")